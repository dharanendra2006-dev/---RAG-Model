"""
Devanagari-safe tokenizer for BM25, now with stopword filtering.
Common function words (है, का, my, the...) appear in nearly every
passage, causing false-positive keyword overlap for off-topic
queries otherwise.
"""
import re

_DELIM_PATTERN = re.compile(r'[\s।,.!?;:"\'()\[\]{}\-–—…]+')

_STOPWORDS = {
    "है", "हैं", "का", "की", "के", "यह", "वह", "ये", "वो", "तो", "भी",
    "से", "में", "को", "पर", "और", "या", "कि", "जो", "न", "नहीं",
    "हो", "था", "थी", "थे", "हुआ", "हुई", "हुए", "कर", "करने", "किया",
    "गया", "गई", "गए", "एक", "कुछ", "सभी", "आप", "हम", "मैं", "मेरे",
    "मेरा", "मेरी", "तुम", "वे", "उस", "इस", "यहाँ", "वहाँ", "कैसे",
    "क्या", "कौन", "कब", "कहाँ", "क्यों", "ना",
    "how", "what", "is", "are", "the", "a", "an", "of", "to", "in",
    "on", "at", "for", "and", "or", "my", "your", "his", "her",
    "their", "this", "that", "was", "were", "be", "been",
}


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = _DELIM_PATTERN.split(text.strip())
    tokens = [t.lower() for t in tokens if t]
    return [t for t in tokens if t not in _STOPWORDS]


if __name__ == "__main__":
    sample = "दिल्ली भारत की राजधानी है। मैनहट्टन परियोजना 1942-1946 में चली।"
    toks = tokenize(sample)
    print(toks)
    assert "दिल्ली" in toks
    assert "है" not in toks
    print("OK - Devanagari tokens intact, stopwords filtered.")
