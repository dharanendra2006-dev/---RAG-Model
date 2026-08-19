"""
Day-1 milestone check: text question -> retrieved passages.
Run this after 04_build_index.py to confirm the index actually works
before wiring it into the FastAPI harness tomorrow.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402

STRATEGY = "sentence"  # try any of: fixed, sentence, semantic, contextual


def main(query: str, k: int = 5):
    model = SentenceTransformer(settings.embedding_model)
    index = faiss.read_index(str(settings.index_dir / f"faiss_{STRATEGY}.index"))
    meta = pd.read_parquet(settings.index_dir / f"meta_{STRATEGY}.parquet")

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, idxs = index.search(q_emb, k)
    print(f"\nQuery: {query}\nStrategy: {STRATEGY}\n")
    for rank, (i, s) in enumerate(zip(idxs[0], scores[0])):
        if i == -1:
            continue
        row = meta.iloc[i]
        print(f"#{rank+1} [score={s:.3f}] chunk_id={row.chunk_id}")
        print(f"    {row.text[:150]}...")


if __name__ == "__main__":
    test_query = sys.argv[1] if len(sys.argv) > 1 else "मैनहट्टन प्रोजेक्ट का प्रभाव क्या था?"
    main(test_query)
