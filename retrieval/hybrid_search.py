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
from transformers import AutoTokenizer

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


def _lazy_load():
    global _session, _hf_tokenizer, _faiss_index, _bm25_data, _meta_df
    if _session is None:
        print(f"[retrieval] loading ONNX embedding model from {ONNX_MODEL_DIR} ...")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(ONNX_MODEL_DIR / "model.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        _hf_tokenizer = AutoTokenizer.from_pretrained(str(ONNX_MODEL_DIR))
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(settings.index_dir / f"faiss_{STRATEGY}.index"))
    if _bm25_data is None:
        with open(settings.index_dir / f"bm25_{STRATEGY}.pkl", "rb") as f:
            _bm25_data = pickle.load(f)
    if _meta_df is None:
        _meta_df = pd.read_parquet(settings.index_dir / f"meta_{STRATEGY}.parquet")
        _meta_df = _meta_df.reset_index(drop=True)


def _encode(texts: list[str]) -> np.ndarray:
    inputs = _hf_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="np",
        return_token_type_ids=True
    )

    input_names = {i.name for i in _session.get_inputs()}
    feed = {k: v for k, v in inputs.items() if k in input_names}

    outputs = _session.run(None, feed)
    last_hidden = outputs[0]

    mask = inputs["attention_mask"][..., None].astype("float32")
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
