"""
Tier 1 extractive answer - ZERO extra embedding calls.

Reuses the dense cosine similarity already computed during retrieval
as the support score, instead of re-embedding candidate sentences.
On CPU, a second small-batch encode call does NOT batch efficiently,
so doing it in the <200ms hot path was the wrong design - this fix
removes it entirely, making extraction near-instant.
"""
import time


def extract_answer(query: str, retrieved: list[dict], model=None) -> dict:
    t0 = time.perf_counter()

    if not retrieved:
        return {
            "text": "",
            "source_chunk_id": None,
            "support_score": 0.0,
            "citations": [],
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
        }

    top = retrieved[0]

    if top.get("dense_score") is not None:
        support_score = float(top["dense_score"])
    else:
        support_score = min(float(top["rrf_score"]) * 10, 0.5)

    return {
        "text": top["text"],
        "source_chunk_id": top["chunk_id"],
        "support_score": support_score,
        "citations": [{"chunk_id": top["chunk_id"], "reason": "top RRF-fused result, dense+sparse agreement"}],
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hybrid_search import hybrid_retrieve

    q = sys.argv[1] if len(sys.argv) > 1 else "मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?"
    retrieval_out = hybrid_retrieve(q)
    ans = extract_answer(q, retrieval_out["results"])

    print(f"\nQuery: {q}")
    print(f"Answer: {ans['text']}")
    print(f"Support score: {ans['support_score']:.4f}")
    print(f"Source: {ans['source_chunk_id']}")
    print(f"Extraction time: {ans['elapsed_ms']:.4f}ms")
