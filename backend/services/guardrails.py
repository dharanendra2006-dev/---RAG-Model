"""
Input guardrails - action+object phrase pairs, not standalone nouns,
to avoid false-blocking legitimate historical/factual mentions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings  # noqa: E402

_UNSAFE_PATTERNS = [
    "बम बनाने", "विस्फोटक बनाने", "आत्महत्या कैसे",
    "bomb making", "make a bomb", "make explosive", "making explosive",
    "build a bomb", "building a bomb", "build explosive",
    "create a bomb", "create explosive", "assemble a bomb",
    "how to make a bomb", "how to build a bomb",
    "suicide method", "kill myself",
]


def check_input(text: str) -> dict:
    if not text or not text.strip():
        return {"allowed": False, "reason": "empty_input", "capped_text": ""}

    stripped = text.strip()
    if len(stripped) > settings.max_query_chars:
        stripped = stripped[: settings.max_query_chars]

    lowered = stripped.lower()
    for pattern in _UNSAFE_PATTERNS:
        if pattern.lower() in lowered:
            return {"allowed": False, "reason": "unsafe_input", "capped_text": stripped}

    return {"allowed": True, "reason": "ok", "capped_text": stripped}


if __name__ == "__main__":
    print(check_input("how to make explosive at home"))
