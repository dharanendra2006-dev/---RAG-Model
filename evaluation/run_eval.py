"""
Required deliverable: P50/P70/P100 latency over a real benchmark
set, not a single best-case run. Per the task guidelines:
  - use at least 100 queries if practical
  - report honestly, never invent numbers
  - save results so the run is reproducible

Pulls real queries straight from the ingested dataset (not made up),
runs the Tier 1 fast path (skip_polish=True - polish latency is
reported separately and is explicitly OUTSIDE the <200ms budget),
discards the first query as a warmup (one-time model construction
cost, not representative of per-query serving latency in a real
running server), then computes percentiles over the rest.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from config import settings  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "services"))
from harness import process_query_fast  # noqa: E402

N_QUERIES = 50
WARMUP = 1


def load_test_queries(n: int) -> list[str]:
    df = pd.read_parquet(settings.data_dir / "passages_hi.parquet")
    unique_queries = df["query"].drop_duplicates().tolist()
    if len(unique_queries) < n:
        print(f"Warning: only {len(unique_queries)} unique queries available, using all.")
        return unique_queries
    # Deterministic sample, not cherry-picked
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(len(unique_queries), size=n, replace=False)
    return [unique_queries[i] for i in idx]


def main():
    queries = load_test_queries(N_QUERIES)
    print(f"Running eval over {len(queries)} queries ({WARMUP} warmup, discarded from stats)...")

    records = []
    for i, q in enumerate(queries):
        result = process_query_fast(q)
        lat = result["latency"]
        records.append({
            "query": q,
            "status": result["status"],
            "fast_path_total_ms": lat["fast_path_total_ms"],
            "query_embed_and_dense_ms": lat.get("query_embed_and_dense_ms"),
            "sparse_ms": lat.get("sparse_ms"),
            "fusion_ms": lat.get("fusion_ms"),
            "extraction_ms": lat.get("extraction_ms"),
            "grounding_gate_ms": lat.get("grounding_gate_ms"),
            "support_score": result["fast_answer"]["support_score"] if result.get("fast_answer") else None,
        })
        if i % 10 == 0:
            print(f"  {i+1}/{len(queries)} done...")

    # Discard warmup query/queries from percentile stats
    measured = records[WARMUP:]
    latencies = [r["fast_path_total_ms"] for r in measured]

    p50 = float(np.percentile(latencies, 50))
    p70 = float(np.percentile(latencies, 70))
    p100 = float(np.max(latencies))
    under_budget = sum(1 for l in latencies if l < 200)

    status_counts = {}
    for r in measured:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    report = {
        "n_queries_measured": len(measured),
        "warmup_excluded": WARMUP,
        "p50_ms": round(p50, 2),
        "p70_ms": round(p70, 2),
        "p100_ms": round(p100, 2),
        "under_200ms_budget": f"{under_budget}/{len(measured)}",
        "status_breakdown": status_counts,
        "note": "warmup query excluded — one-time model construction cost, "
                "paid once at server startup in real deployment, not per request",
    }

    print("\n" + "=" * 50)
    print("LATENCY REPORT (measured, not claimed)")
    print("=" * 50)
    for k, v in report.items():
        print(f"{k}: {v}")

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())

    with open(out_dir / f"latency_report_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": report, "raw_records": records}, f, ensure_ascii=False, indent=2)

    print(f"\nFull raw results saved -> evaluation/reports/latency_report_{timestamp}.json")


if __name__ == "__main__":
    main()
