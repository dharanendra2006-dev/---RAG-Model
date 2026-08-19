"""
Guardrail test harness — required checklist items: off-topic
handling, unsafe-input handling, abstention, malformed input. Runs
each categorized test query through the REAL harness, compares
actual status against expected, reports pass/fail per category, and
saves full results for reproducibility (same "measured, not
claimed" standard as the latency eval).
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "services"))
from harness import process_query_fast  # noqa: E402

TEST_FILE = Path(__file__).resolve().parent / "test_queries.json"


def main():
    with open(TEST_FILE, encoding="utf-8") as f:
        tests = json.load(f)

    print(f"Running {len(tests)} guardrail test queries...\n")

    results = []
    for t in tests:
        query = t["query"]
        if t.get("repeat_to_exceed_cap"):
            query = (query * 60)[:600]  # push past the 512-char cap deliberately

        result = process_query_fast(query)
        actual = result["status"]
        expected = t["expected_status"]
        passed = actual == expected

        # support_score used to be read only from fast_answer, which is
        # None on abstain/block - that silently dropped the real dense
        # score for every rejected query, which is exactly the number
        # needed to tune the grounding gate. grounding.score carries the
        # same value (set by extract_answer) regardless of pass/fail, so
        # fall back to that. retrieved[0] gives dense_score/sparse_rank
        # directly for full diagnostic visibility even when blocked.
        if result.get("fast_answer"):
            support_score = result["fast_answer"]["support_score"]
        elif result.get("grounding"):
            support_score = result["grounding"]["score"]
        else:
            support_score = None

        top = result["retrieved"][0] if result.get("retrieved") else {}

        results.append({
            "category": t["category"],
            "query": query[:80] + ("..." if len(query) > 80 else ""),
            "expected_status": expected,
            "actual_status": actual,
            "passed": passed,
            "guardrail_flag": result.get("guardrail_flag"),
            "support_score": support_score,
            "top_dense_score": top.get("dense_score"),
            "top_sparse_rank": top.get("sparse_rank"),
            "message": result.get("message"),
            "note": t.get("note"),
        })

        icon = "PASS" if passed else "FAIL"
        print(f"[{icon}] {t['category']:<24} expected={expected:<14} actual={actual:<14} "
              f"dense={top.get('dense_score')} sparse_rank={top.get('sparse_rank')}")

    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"RESULT: {n_passed}/{len(results)} passed")
    print(f"{'='*60}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFAILED CASES (investigate these):")
        for r in failed:
            print(f"  - {r['category']}: expected={r['expected_status']} got={r['actual_status']} "
                  f"(support={r['support_score']}) — {r['note']}")

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(exist_ok=True)
    timestamp = int(time.time())
    out_path = out_dir / f"guardrail_report_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_passed": n_passed, "n_total": len(results), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved -> {out_path}")


if __name__ == "__main__":
    main()
