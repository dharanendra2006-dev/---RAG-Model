"""
Required deliverable: real P50/P70/P100 latency over a batch of
queries, run against the LIVE deployed API.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://rag-model-production-3de3.up.railway.app"
N_QUERIES = 50
WARMUP = 1

FALLBACK_QUERIES = [
    "???????? ???????? ?? ????? ?? ?????? ???? ???",
    "?????? ?? ????? ??????",
    "???????? ???????? ?? ???? ????",
    "??????? ????? ????? ?? ?????? ????",
    "?????? ????? ?? ?????????? ????? ???? ???",
    "???????? ???????? 1942 1946 ??????",
    "?????? ?????? ?? ?????? ???? ???",
    "???????? ????????? ?? ???????? ???? ???",
]


def load_queries(n: int) -> list[str]:
    # Disk space constraint on this machine - pyarrow won't install,
    # skip parquet loading and go straight to the curated fallback set.
    print(f"Using {len(FALLBACK_QUERIES)} curated real Hindi queries, repeated to reach {n}")
    return [FALLBACK_QUERIES[i % len(FALLBACK_QUERIES)] for i in range(n)]


def _unused_load_queries(n: int) -> list[str]:
    for candidate in [Path("data/passages_hi.parquet"), Path("../data/passages_hi.parquet")]:
        if candidate.exists():
            import pandas as pd
            df = pd.read_parquet(candidate)
            unique_queries = df["query"].drop_duplicates().tolist()
            if len(unique_queries) >= n:
                rng = np.random.default_rng(seed=42)
                idx = rng.choice(len(unique_queries), size=n, replace=False)
                print(f"Loaded {n} real queries from {candidate}")
                return [unique_queries[i] for i in idx]

    print(f"No local dataset found - using {len(FALLBACK_QUERIES)} curated queries, repeated to reach {n}")
    return [FALLBACK_QUERIES[i % len(FALLBACK_QUERIES)] for i in range(n)]


def percentiles(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p70": round(float(np.percentile(arr, 70)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
        "p100": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
    }


def main():
    queries = load_queries(N_QUERIES)
    print(f"\nRunning {len(queries)} queries against {BASE_URL} ({WARMUP} warmup, excluded from stats)...\n")

    records = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE_URL}/api/query",
                json={"text": q, "mode": "fast"},
                timeout=20,
            )
            round_trip_ms = (time.perf_counter() - t0) * 1000
            data = resp.json()
            fast_path_ms = data.get("latency", {}).get("fast_path_total_ms")
            status = data.get("status")
        except Exception as e:
            round_trip_ms = (time.perf_counter() - t0) * 1000
            fast_path_ms = None
            status = f"ERROR: {e}"

        records.append({
            "query": q,
            "status": status,
            "fast_path_total_ms": fast_path_ms,
            "round_trip_ms": round_trip_ms,
        })
        if i % 10 == 0:
            print(f"  {i+1}/{len(queries)} done...")

    measured = records[WARMUP:]
    fast_path_values = [r["fast_path_total_ms"] for r in measured if r["fast_path_total_ms"] is not None]
    round_trip_values = [r["round_trip_ms"] for r in measured]

    status_counts = {}
    for r in measured:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    report = {
        "base_url": BASE_URL,
        "n_queries_measured": len(measured),
        "warmup_excluded": WARMUP,
        "server_side_pipeline_latency_ms": percentiles(fast_path_values) if fast_path_values else {},
        "network_inclusive_round_trip_ms": percentiles(round_trip_values) if round_trip_values else {},
        "status_breakdown": status_counts,
        "under_200ms_budget_pipeline": (
            f"{sum(1 for v in fast_path_values if v < 200)}/{len(fast_path_values)}" if fast_path_values else "n/a"
        ),
        "note": (
            "server_side_pipeline_latency_ms is the number the <200ms budget "
            "requirement is about. network_inclusive_round_trip_ms adds "
            "internet + hosting-platform overhead on top - reported separately."
        ),
    }

    print("\n" + "=" * 60)
    print("LIVE DEPLOYMENT LATENCY REPORT (measured, not claimed)")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"live_latency_report_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": report, "raw_records": records}, f, ensure_ascii=False, indent=2)
    print(f"\nFull raw results saved -> {out_path}")


if __name__ == "__main__":
    main()
