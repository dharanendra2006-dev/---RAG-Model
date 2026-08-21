"""
Tier 1 harness: process_query_fast() runs guardrail -> hybrid
retrieval -> extraction -> grounding gate, all inside the measured
<200ms budget.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402
from services.guardrails import check_input  # noqa: E402
from services.grounding import grounding_gate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "retrieval"))
from hybrid_search import hybrid_retrieve, _lazy_load  # noqa: E402
from extractive import extract_answer  # noqa: E402


def process_query_fast(text: str) -> dict:
    t_start = time.perf_counter()
    latency = {}

    t0 = time.perf_counter()
    guard = check_input(text)
    latency["guardrail_ms"] = (time.perf_counter() - t0) * 1000

    if not guard["allowed"]:
        return {
            "status": "blocked",
            "guardrail_flag": guard["reason"],
            "message": f"Blocked: {guard['reason']}",
            "fast_answer": None,
            "retrieved": [],
            "grounding": None,
            "latency": {**latency, "fast_path_total_ms": (time.perf_counter() - t_start) * 1000},
        }

    query_text = guard["capped_text"]

    _lazy_load()
    t0 = time.perf_counter()
    retrieved = hybrid_retrieve(query_text)
    latency["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    ans = extract_answer(query_text, retrieved)
    latency["extraction_ms"] = ans["elapsed_ms"]

    t0 = time.perf_counter()
    has_sparse_hit = bool(retrieved) and retrieved[0].get("sparse_rank") is not None
    gate = grounding_gate(ans["support_score"], has_results=bool(retrieved), has_sparse_hit=has_sparse_hit)
    latency["grounding_gate_ms"] = (time.perf_counter() - t0) * 1000

    latency["fast_path_total_ms"] = (time.perf_counter() - t_start) * 1000

    if not gate["passed"]:
        return {
            "status": "abstained",
            "guardrail_flag": "insufficient_evidence",
            "message": "I don't have enough information in the provided corpus to answer that.",
            "fast_answer": None,
            "retrieved": retrieved,
            "grounding": {"supported": False, "score": ans["support_score"], "unsupported_claims": []},
            "latency": latency,
        }

    return {
        "status": "answered_fast",
        "guardrail_flag": "none",
        "message": None,
        "fast_answer": {
            "text": ans["text"],
            "source_chunk_id": ans["source_chunk_id"],
            "support_score": ans["support_score"],
            "citations": ans["citations"],
        },
        "retrieved": retrieved,
        "grounding": {"supported": True, "score": ans["support_score"], "unsupported_claims": []},
        "latency": latency,
    }


if __name__ == "__main__":
    test_queries = [
        "???????? ???????? ?? ????? ?? ?????? ???? ???",
        "?? ???? ???? ???",
    ]
    for q in test_queries:
        print(f"\n{'='*60}\nQuery: {q}")
        result = process_query_fast(q)
        print(f"Status: {result['status']}")
        if result["fast_answer"]:
            print(f"Answer: {result['fast_answer']['text']}")
            print(f"Support: {result['fast_answer']['support_score']:.4f}")
        else:
            print(f"Message: {result['message']}")
        print(f"Fast path total: {result['latency']['fast_path_total_ms']:.2f}ms")
        print(f"Stage breakdown: {result['latency']}")


from services.polish import polish_answer, verify_polish  # noqa: E402


def process_query(text: str, skip_polish: bool = False) -> dict:
    fast_result = process_query_fast(text)

    if skip_polish or fast_result["status"] != "answered_fast":
        fast_result["latency"]["end_to_end_ms"] = fast_result["latency"]["fast_path_total_ms"]
        fast_result["polished_answer"] = None
        return fast_result



    fast_answer_text = fast_result["fast_answer"]["text"]
    top_chunk = fast_result["retrieved"][0]["text"] if fast_result["retrieved"] else fast_answer_text

    polish_out = polish_answer(text, fast_answer_text, top_chunk)
    fast_result["latency"]["polish_ms"] = polish_out["elapsed_ms"]

    if polish_out["error"] or not polish_out["text"]:
        fast_result["polished_answer"] = {
            "text": None, "model_used": settings.generation_model,
            "verified": False, "rejection_reason": polish_out["error"] or "empty_output",
        }
        fast_result["status"] = "answered_fast"
        fast_result["latency"]["end_to_end_ms"] = (
            fast_result["latency"]["fast_path_total_ms"] + polish_out["elapsed_ms"]
        )
        return fast_result

    verify_out = verify_polish(polish_out["text"], fast_answer_text)
    fast_result["latency"]["verify_ms"] = verify_out["elapsed_ms"]

    if verify_out["verified"]:
        fast_result["status"] = "answered_polished"
        fast_result["polished_answer"] = {
            "text": polish_out["text"], "model_used": polish_out["model_used"],
            "verified": True, "rejection_reason": None,
        }
    else:
        fast_result["status"] = "answered_fast"
        fast_result["polished_answer"] = {
            "text": polish_out["text"], "model_used": polish_out["model_used"],
            "verified": False, "rejection_reason": verify_out["reason"],
        }

    fast_result["latency"]["end_to_end_ms"] = (
        fast_result["latency"]["fast_path_total_ms"] + polish_out["elapsed_ms"] + verify_out["elapsed_ms"]
    )
    return fast_result
