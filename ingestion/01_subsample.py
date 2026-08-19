"""
Reads the already-downloaded Hindi shard (train/hintrain.parquet)
in small batches via pyarrow directly, bypassing the `datasets`
library's full-file arrow-conversion step (which needs a single
large contiguous memory allocation and crashes on lower-RAM
machines). Stops as soon as TARGET_QUERY_ROWS is reached, so most
of the file is never even read into memory.
"""
import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402

REPO = settings.hf_dataset_repo
TARGET_QUERY_ROWS = settings.target_query_rows
OUT_PATH = settings.data_dir / "passages_hi.parquet"
HINDI_PATH = "train/hintrain.parquet"
BATCH_SIZE = 500


def flatten_row(ex):
    out = []
    query = ex.get("query") or ex.get("Query") or ""
    query_id = ex.get("query_id", None)
    query_type = ex.get("query_type", "UNKNOWN")
    passages = ex.get("passages", {}) or {}

    eng_passages = passages.get("English_passages", []) or []
    trans_passages = passages.get("Translated_passages", []) or []
    is_selected = passages.get("is_selected", []) or []

    texts = trans_passages if trans_passages else eng_passages
    if not texts or not query:
        return out

    for i, text in enumerate(texts):
        if not text or not text.strip():
            continue
        sel = is_selected[i] if i < len(is_selected) else 0
        out.append({
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "passage_idx": i,
            "document_id": f"{query_id}_{i}",
            "passage_text": text.strip(),
            "is_selected": int(sel),
            "lang": "hin",
        })
    return out


def main():
    print("Locating cached file (already downloaded, no re-fetch)...")
    local_path = hf_hub_download(
        repo_id=REPO, repo_type="dataset", filename=HINDI_PATH
    )
    print(f"Using local file: {local_path}")

    pf = pq.ParquetFile(local_path)
    print(f"Total rows in shard: {pf.metadata.num_rows}, row groups: {pf.num_row_groups}")

    rows = []
    skipped = 0
    n_source_seen = 0

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        batch_dicts = batch.to_pylist()
        for ex in batch_dicts:
            if n_source_seen == 0:
                print("First row keys:", list(ex.keys()))
            n_source_seen += 1
            try:
                rows.extend(flatten_row(ex))
            except Exception:
                skipped += 1
            if n_source_seen >= TARGET_QUERY_ROWS:
                break
        print(f"processed {n_source_seen}/{TARGET_QUERY_ROWS} source rows -> {len(rows)} chunk rows so far")
        if n_source_seen >= TARGET_QUERY_ROWS:
            break

    print(f"Done. Total passage rows: {len(rows)}, skipped malformed rows: {skipped}")
    if not rows:
        raise RuntimeError("No rows collected — check the 'First row keys' printed above against flatten_row().")

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["query_id", "passage_idx"], inplace=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows -> {OUT_PATH}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
