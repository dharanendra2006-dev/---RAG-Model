"""
Grounding gate - requires either REAL lexical corroboration (BM25
hit) or a very high dense-only score. Plain dense cosine at the
normal threshold was proven unreliable by run_guardrail_tests.py -
off-topic queries scored 0.79-0.85, indistinguishable from real
matches at min_groundedness_score. Requiring a BM25 hit unconditionally
fixed that, but also rejects genuine synonym paraphrases that share
no keywords with the source text at all (semantic_paraphrase test
case). settings.dense_bypass_score is a second, higher bar: only a
dense score that clears BOTH the false-positive ceiling AND has real
margin below it is allowed through without lexical corroboration.
See config.py for the provisional-value caveat on that threshold.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402


def grounding_gate(support_score: float, has_results: bool, has_sparse_hit: bool = None) -> dict:
    if not has_results:
        return {"passed": False, "reason": "no_retrieval_results"}

    if support_score < settings.min_groundedness_score:
        return {
            "passed": False,
            "reason": f"support_score {support_score:.3f} below threshold {settings.min_groundedness_score}",
        }

    if has_sparse_hit is False:
        if support_score >= settings.dense_bypass_score:
            return {
                "passed": True,
                "reason": (
                    f"dense-only match (support={support_score:.3f}) clears bypass "
                    f"threshold {settings.dense_bypass_score} - treated as a real "
                    f"semantic match despite no keyword overlap"
                ),
            }
        return {
            "passed": False,
            "reason": (
                f"dense-only match (support={support_score:.3f}) below bypass "
                f"threshold {settings.dense_bypass_score}, no real keyword overlap"
            ),
        }

    return {"passed": True, "reason": f"support_score {support_score:.3f} meets threshold, keyword overlap confirmed"}


if __name__ == "__main__":
    print(grounding_gate(0.65, True, has_sparse_hit=True))
    print(grounding_gate(0.90, True, has_sparse_hit=False))  # above bypass -> pass
    print(grounding_gate(0.85, True, has_sparse_hit=False))  # below bypass -> fail
    print(grounding_gate(0.20, True, has_sparse_hit=True))
    print(grounding_gate(0.0, False))
