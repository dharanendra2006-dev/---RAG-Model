"""
Runs test_queries.json against the LIVE deployed API (not local
process_query_fast) - exercises the real deployment end to end.
"""
import json
import sys
import time
from pathlib import Path
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://rag-model-production-3de3.up.railway.app"
TEST_FILE = Path(__file__).resolve().parent / "test_queries.json"


def main():
    with open(TEST_FILE, encoding="utf-8") as f:
        tests = json.load(f)

    print(f"Running {len(tests)} guardrail test queries against {BASE_URL} ...\n")

    results = []
    for t in tests:
        query = t["query"]
        if t.get("repeat_to_exceed_cap"):
            query = (query * 60)[:600]

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{BASE_URL}/api/query",
                json={"text": query, "mode": "fast"},
                timeout=15,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            data = resp.json()
            actual = data.get("status", f"HTTP_{resp.status_code}")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            actual = f"REQUEST_ERROR: {e}"
            data = {}

        expected = t["expected_status"]
        passed = actual == expected

        results.append({
            "category": t["category"],
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "elapsed_ms": elapsed_ms,
            "support_score": (data.get("fast_answer") or {}).get("support_score"),
            "note": t.get("note"),
        })

        icon = "PASS" if passed else "FAIL"
        print(f"[{icon}] {t['category']:<24} expected={expected:<14} actual={actual:<14} ({elapsed_ms:.0f}ms)  {t.get('note','')}")

    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"RESULT: {n_passed}/{len(results)} passed")
    print(f"{'='*60}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFAILED CASES:")
        for r in failed:
            print(f"  - {r['category']}: expected={r['expected']} got={r['actual']} (support={r['support_score']}) - {r['note']}")

    latencies = [r["elapsed_ms"] for r in results]
    latencies.sort()
    if latencies:
        def pct(p):
            idx = min(int(len(latencies) * p / 100), len(latencies) - 1)
            return latencies[idx]
        print(f"\nLive HTTP latency (includes network round-trip):")
        print(f"  P50: {pct(50):.0f}ms  P70: {pct(70):.0f}ms  P100: {pct(100):.0f}ms")


if __name__ == "__main__":
    main()
