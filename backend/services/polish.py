"""
Tier 2: LLM polish + verify. Runs OUTSIDE the <200ms budget, wraps
the Tier 1 fast answer, never runs before it.

Polish step: Claude rewords the extractive answer more naturally,
explicitly instructed not to add facts beyond the given context.
Verify step: check the polished text stays close (embedding
similarity) to the ORIGINAL fast answer / source chunk. If the
polish drifts too far (possible hallucination) or the API call
fails/times out, fall back to the fast answer untouched. Polish is
strictly additive - it can never replace grounding.
"""
import sys
import time
import logging
from pathlib import Path

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

logger = logging.getLogger("voice-rag.polish")

_client = None


def _lazy_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _retryable_anthropic_exceptions():
    """Import lazily so this module doesn't hard-depend on `anthropic`
    at import time. Only retry errors that are genuinely transient -
    a bad API key or malformed request will fail identically on every
    attempt, so retrying those just wastes time on top of the already
    "outside budget" polish step."""
    import anthropic
    return (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.OverloadedError,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception(lambda e: not isinstance(e, _NonRetryable)),
    reraise=True,
)
def _create_with_retry(client, **kwargs):
    try:
        return client.messages.create(**kwargs)
    except _retryable_anthropic_exceptions() as e:
        logger.warning("Transient Anthropic API error, will retry: %s: %s", type(e).__name__, e)
        raise
    except Exception as e:
        # Non-transient (auth, bad request, etc.) - fail immediately,
        # don't burn retry attempts on something that can't succeed.
        # Chained via `from e` so the real underlying error is still
        # visible in the traceback/str(), not swallowed by the wrapper.
        raise _NonRetryable(str(e)) from e


class _NonRetryable(Exception):
    pass


def polish_answer(query: str, fast_answer_text: str, source_text: str) -> dict:
    """
    Returns {"text": str, "model_used": str, "elapsed_ms": float,
    "error": str|None}. Never raises - a failure here should never
    break the response, since the fast answer is always the fallback.
    Transient API errors (connection drop, rate limit, 5xx) are
    retried up to 3x with exponential backoff before giving up.
    """
    t0 = time.perf_counter()

    if not settings.anthropic_api_key:
        return {"text": None, "model_used": None, "elapsed_ms": 0.0, "error": "no_api_key_configured"}

    try:
        client = _lazy_client()
        system = (
            "You reword a given answer to sound more natural, in the SAME language "
            "as the question. You must NOT add any fact, number, name, or claim that "
            "is not already present in the provided answer or source text. If you "
            "cannot reword without adding new information, return the original answer "
            "text unchanged. Output ONLY the reworded answer, nothing else."
        )
        user_msg = (
            f"Question: {query}\n\n"
            f"Answer to reword (do not add new facts): {fast_answer_text}\n\n"
            f"Source context (for reference only): {source_text[:500]}"
        )
        try:
            resp = _create_with_retry(
                client,
                model=settings.generation_model,
                max_tokens=settings.max_answer_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                timeout=settings.stage_timeout_s,
            )
        except _NonRetryable:
            # Unwrap back to the real underlying exception for an
            # accurate error message below.
            raise
        text = "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
        return {
            "text": text,
            "model_used": settings.generation_model,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
            "error": None,
        }
    except Exception as e:
        return {
            "text": None,
            "model_used": settings.generation_model,
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
            "error": str(e),
        }


def verify_polish(polished_text: str, fast_answer_text: str, threshold: float = None) -> dict:
    """
    Embedding-similarity check between the polished text and the
    ORIGINAL fast (grounded) answer. Cheap reuse of the already-
    loaded embedding model - no extra API call needed for this.
    """
    import numpy as np
    from hybrid_search import _encode
    t0 = time.perf_counter()
    threshold = threshold or settings.min_groundedness_score

    if not polished_text:
        return {"verified": False, "reason": "empty_polish_output", "elapsed_ms": (time.perf_counter() - t0) * 1000}

    embs = _encode(["passage: " + polished_text, "passage: " + fast_answer_text])
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim = float(embs[0] @ embs[1])

    elapsed = (time.perf_counter() - t0) * 1000
    if sim < threshold:
        return {"verified": False, "reason": f"drift from fast answer, sim={sim:.3f} < {threshold}", "elapsed_ms": elapsed}
    return {"verified": True, "reason": f"sim={sim:.3f} >= {threshold}", "elapsed_ms": elapsed}


if __name__ == "__main__":
    q = "मैनहट्टन परियोजना की सफलता का प्रभाव क्या था?"
    fast = "मैनहट्टन प्रोजेक्ट का उद्देश्य यह देखना था कि परमाणु बम बनाना संभव है या नहीं।"
    result = polish_answer(q, fast, fast)
    print("Polish result:", result)
    if result["text"]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.embedding_model)
        v = verify_polish(result["text"], fast, model)
        print("Verify result:", v)
