"""
Reads data/passages_hi.parquet, runs all chunk views, writes
data/chunks_all_views.parquet. Uses a small fast multilingual model
(paraphrase-multilingual-MiniLM, ~470MB) for semantic-boundary
detection instead of bge-m3 — that model is reserved for the real
retrieval embeddings in 04_build_index.py, where quality matters
more than speed. This alone should cut runtime from hours to
minutes.
"""
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402
from importlib import import_module
chunking = import_module("02_chunking")

IN_PATH = settings.data_dir / "passages_hi.parquet"
OUT_PATH = settings.data_dir / "chunks_all_views.parquet"
SEMANTIC_HELPER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_embed_fn():
    from sentence_transformers import SentenceTransformer
    print(f"Loading lightweight model for semantic chunking: {SEMANTIC_HELPER_MODEL}")
    model = SentenceTransformer(SEMANTIC_HELPER_MODEL)
    return lambda sentences: model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)


def main():
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df)} source passages from {IN_PATH}")

    embed_fn = load_embed_fn()
    all_records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row["passage_text"]
        doc_id = row["document_id"]
        lang = row["lang"]
        qtype = row.get("query_type")
        sel = row.get("is_selected")

        fixed = chunking.fixed_chunk(text, doc_id, settings.fixed_chunk_size, settings.fixed_chunk_overlap)
        sentence = chunking.sentence_chunk(text, doc_id, settings.sentence_chunk_target_size)
        try:
            semantic = chunking.semantic_chunk(text, doc_id, embed_fn, settings.semantic_similarity_threshold)
        except Exception:
            semantic = sentence
        contextual = chunking.contextual_chunk(text, doc_id, fixed)

        all_records += chunking.metadata_aware_records(fixed, doc_id, "fixed", lang, qtype, sel)
        all_records += chunking.metadata_aware_records(sentence, doc_id, "sentence", lang, qtype, sel)
        all_records += chunking.metadata_aware_records(semantic, doc_id, "semantic", lang, qtype, sel)
        all_records += chunking.metadata_aware_records(
            [{"idx": c["idx"], "text": c["parent_text"]} for c in contextual],
            doc_id, "contextual", lang, qtype, sel,
        )

    out_df = pd.DataFrame(all_records)
    out_df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} chunk records -> {OUT_PATH}")
    print(out_df.groupby("strategy").size())


if __name__ == "__main__":
    main()
