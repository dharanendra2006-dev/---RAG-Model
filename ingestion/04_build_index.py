"""
Offline step: embed every chunk once, build a FAISS index PER
STRATEGY (so retrieval can query one view or ensemble across all),
and save both the index and the metadata table needed to map
FAISS row -> chunk_id/text back at runtime.

Never re-run this per query — runtime only embeds the question.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402

CHUNKS_PATH = settings.data_dir / "chunks_all_views.parquet"


def main():
    df = pd.read_parquet(CHUNKS_PATH)
    print(f"Loaded {len(df)} chunks across strategies: {df.strategy.unique().tolist()}")

    model = SentenceTransformer(settings.embedding_model)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    for strategy, group in df.groupby("strategy"):
        texts = ["passage: " + t for t in group["text"].tolist()]
        print(f"Embedding {len(texts)} chunks for strategy={strategy} ...")
        embs = model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        faiss.normalize_L2(embs)  # cosine similarity via inner product
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        index_path = settings.index_dir / f"faiss_{strategy}.index"
        faiss.write_index(index, str(index_path))

        meta_path = settings.index_dir / f"meta_{strategy}.parquet"
        group.reset_index(drop=True).to_parquet(meta_path, index=False)

        print(f"  saved {index_path.name}, {meta_path.name}")

    # Also dump embedding dim + model name so retrieval.py can sanity-check
    manifest = {
        "embedding_model": settings.embedding_model,
        "dim": int(embs.shape[1]),
        "strategies": df.strategy.unique().tolist(),
    }
    with open(settings.index_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Done. manifest.json written.")


if __name__ == "__main__":
    main()
