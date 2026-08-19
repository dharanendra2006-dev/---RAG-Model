"""
Builds a BM25 index, also saving the tokenized corpus so retrieval
can compute REAL token-overlap counts at query time.
"""
import sys
import pickle
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402
from tokenizer import tokenize  # noqa: E402

STRATEGY = settings.served_strategy


def main():
    meta_path = settings.index_dir / f"meta_{STRATEGY}.parquet"
    df = pd.read_parquet(meta_path)
    print(f"Building BM25 over {len(df)} chunks (strategy={STRATEGY})")

    tokenized_corpus = [tokenize(t) for t in df["text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)

    out = {
        "bm25": bm25,
        "chunk_ids": df["chunk_id"].tolist(),
        "tokenized_corpus": tokenized_corpus,
    }
    out_path = settings.index_dir / f"bm25_{STRATEGY}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved BM25 index -> {out_path}")


if __name__ == "__main__":
    main()
