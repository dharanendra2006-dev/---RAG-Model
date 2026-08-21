import os
os.environ["HF_HUB_OFFLINE"] = "1"
"""
Tier 1 retrieval: dense (FAISS) + sparse (BM25) search over the
served chunk strategy, combined via Reciprocal Rank Fusion (RRF).
"""
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import onnxruntime as ort
from tokenizers import Tokenizer
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import tokenize  # noqa: E402

STRATEGY = settings.served_strategy
ONNX_MODEL_DIR = Path(__file__).resolve().parent.parent / "backend" / "models" / "e5-small-onnx"

_session = None
_hf_tokenizer = None
_faiss_index = None
_bm25_data = None
_meta_df = None


def _mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        # Windows has no `resource` module - fall back to psutil for
        # local testing. Render (Linux) always uses the branch above.
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return -1.0


def _lazy_load():
    global _session, _hf_tokenizer, _faiss_index, _bm25_data, _meta_df
    print(f"[MEM] _lazy_load start: {_mem_mb():.0f} MB")
    if _session is None:
        print(f"[retrieval] loading ONNX embedding model from {ONNX_MODEL_DIR} ...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _session = ort.InferenceSession(
            str(ONNX_MODEL_DIR / "model_quantized.onnx"),
            sess_options=so,
            providers=[("CPUExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
        )
        print(f"[MEM] after ONNX session: {_mem_mb():.0f} MB")
        _hf_tokenizer = Tokenizer.from_file(str(ONNX_MODEL_DIR / "tokenizer.json"))
        _special = json.loads((ONNX_MODEL_DIR / "special_tokens_map.json").read_text(encoding="utf-8"))
        _pad_token = _special.get("pad_token")
        if isinstance(_pad_token, dict):
            _pad_token = _pad_token.get("content")
        _pad_token = _pad_token or "<pad>"
        _pad_id = _hf_tokenizer.token_to_id(_pad_token) or 0
        _hf_tokenizer.enable_padding(pad_id=_pad_id, pad_token=_pad_token)
        _hf_tokenizer.enable_truncation(max_length=512)
        print(f"[MEM] after tokenizer: {_mem_mb():.0f} MB")
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(settings.index_dir / f"faiss_{STRATEGY}.index"))
        print(f"[MEM] after faiss index: {_mem_mb():.0f} MB")
    if _bm25_data is None:
        with open(settings.index_dir / f"bm25_{STRATEGY}.pkl", "rb") as f:
            _bm25_data = pickle.load(f)
        print(f"[MEM] after bm25: {_mem_mb():.0f} MB")
    if _meta_df is None:
        _meta_df = pd.read_parquet(settings.index_dir / f"meta_{STRATEGY}.parquet")
        _meta_df = _meta_df.reset_index(drop=True)
        print(f"[MEM] after meta parquet: {_mem_mb():.0f} MB")


def _encode(texts: list[str]) -> np.ndarray:
    encs = _hf_tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encs], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encs], dtype=np.int64)

    feed_all = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    input_names = {i.name for i in _session.get_inputs()}
    feed = {k: v for k, v in feed_all.items() if k in input_names}

    outputs = _session.run(None, feed)
    print(f"[MEM] after _session.run: {_mem_mb():.0f} MB")
    last_hidden = outputs[0]

    mask = attention_mask[..., None].astype("float32")
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return (summed / counts).astype("float32")


def dense_search(query: str, k: int) -> tuple[list[int], list[float], float]:
    t0 = time.perf_counter()
    _lazy_load()
    q_emb = _encode(["query: " + query])
    faiss.normalize_L2(q_emb)
    scores, idxs = _faiss_index.search(q_emb, k)
    elapsed = (time.perf_counter() - t0) * 1000

    valid = [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
    rows = [v[0] for v in valid]
    scs = [v[1] for v in valid]
    return rows, scs, elapsed


def sparse_search(query: str, k: int) -> tuple[list[int], list[float], float]:
    t0 = time.perf_counter()
    _lazy_load()
    tokens = tokenize(query)
    query_token_set = set(tokens)

    if not query_token_set:
        min_overlap = 0
    elif len(query_token_set) <= 3:
        min_overlap = len(query_token_set)
    else:
        min_overlap = 3

    if hasattr(_bm25_data, "get_scores"):
        scores = _bm25_data.get_scores(tokens)
    else:
        scores = np.zeros(len(_meta_df))
        for token in query_token_set:
            if token in _bm25_data:
                for doc_id, score in _bm25_data[token].items():
                    scores[doc_id] += score

    top_indices = np.argsort(scores)[::-1][:k]
    rows = [int(i) for i in top_indices if scores[i] > 0]
    scs = [float(scores[i]) for i in rows]
    elapsed = (time.perf_counter() - t0) * 1000
    return rows, scs, elapsed


def hybrid_retrieve(query: str, top_k: int = 5) -> list[dict]:
    _lazy_load()
    v_k = getattr(settings, "vector_top_k", top_k * 3)

    d_rows, _, _ = dense_search(query, k=v_k)
    s_rows, _, _ = sparse_search(query, k=v_k)

    rrf_scores = {}
    for rank, idx in enumerate(d_rows):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60.0 + rank))
    for rank, idx in enumerate(s_rows):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (60.0 + rank))

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in sorted_docs:
        row_data = _meta_df.iloc[idx].to_dict()
        row_data["rrf_score"] = float(score)
        results.append(row_data)

    return results

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "test query"
    print(hybrid_retrieve(q, top_k=2))
