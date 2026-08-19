"""
Five chunk views over the same passage corpus. Each returns a list
of dicts matching backend/models/schemas.py::Chunk so they can be
embedded and indexed uniformly, tagged by `strategy`.

MS MARCO passages are already short, coherent units (usually one
paragraph), so "fixed/sentence/semantic" mostly matter when a
passage is long; contextual and metadata-aware are cheap wins that
apply regardless of passage length.
"""
import re
import uuid
from typing import Optional


def _cid(doc_id: str, strategy: str, idx: int) -> str:
    return f"{doc_id}::{strategy}::{idx}"


def fixed_chunk(text: str, doc_id: str, size: int = 256, overlap: int = 50) -> list[dict]:
    """View 1 — baseline. Fixed word-count windows with overlap."""
    words = text.split()
    if len(words) <= size:
        return [{"idx": 0, "text": text}]
    chunks = []
    step = max(size - overlap, 1)
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start:start + size]
        if not window:
            break
        chunks.append({"idx": i, "text": " ".join(window)})
        if start + size >= len(words):
            break
    return chunks


def sentence_chunk(text: str, doc_id: str, target_size: int = 200) -> list[dict]:
    """View 2 — split on sentence boundaries, pack sentences up to target_size words."""
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())  # ।  = Hindi danda
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return [{"idx": 0, "text": text}]

    chunks, current, current_len = [], [], 0
    idx = 0
    for sent in sentences:
        wc = len(sent.split())
        if current and current_len + wc > target_size:
            chunks.append({"idx": idx, "text": " ".join(current)})
            idx += 1
            current, current_len = [], 0
        current.append(sent)
        current_len += wc
    if current:
        chunks.append({"idx": idx, "text": " ".join(current)})
    return chunks


def semantic_chunk(text: str, doc_id: str, embed_fn, threshold: float = 0.55) -> list[dict]:
    """
    View 3 — split where consecutive sentence embeddings diverge
    (topic-shift detection). embed_fn: callable(list[str]) -> np.ndarray.
    Falls back to sentence_chunk if too short to bother.
    """
    import numpy as np

    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= 2:
        return sentence_chunk(text, doc_id)

    embs = embed_fn(sentences)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    chunks, current, idx = [], [sentences[0]], 0
    for i in range(1, len(sentences)):
        sim = float(np.dot(embs[i - 1], embs[i]))
        if sim < threshold and current:
            chunks.append({"idx": idx, "text": " ".join(current)})
            idx += 1
            current = []
        current.append(sentences[i])
    if current:
        chunks.append({"idx": idx, "text": " ".join(current)})
    return chunks


def contextual_chunk(text: str, doc_id: str, base_chunks: list[dict]) -> list[dict]:
    """
    View 4 — wraps chunks from any base strategy with parent context
    (the full original passage) so retrieval isn't working with
    isolated fragments. `parent_id` links back to the source passage.
    """
    out = []
    for c in base_chunks:
        out.append({
            "idx": c["idx"],
            "text": c["text"],
            "parent_text": text,
        })
    return out


def metadata_aware_records(
    base_chunks: list[dict],
    doc_id: str,
    strategy: str,
    language: str,
    query_type: Optional[str] = None,
    is_selected: Optional[int] = None,
    parent_id: Optional[str] = None,
) -> list[dict]:
    """View 5 — attach full metadata; produces final Chunk-shaped records."""
    records = []
    for c in base_chunks:
        records.append({
            "chunk_id": _cid(doc_id, strategy, c["idx"]),
            "document_id": doc_id,
            "parent_id": parent_id or doc_id,
            "language": language,
            "strategy": strategy,
            "text": c["text"],
            "query_type": query_type,
            "is_selected_gt": is_selected,
        })
    return records


if __name__ == "__main__":
    sample = ("The immediate impact of the success of the Manhattan Project was "
               "the creation of the atomic bomb. This changed warfare forever. "
               "Scientists later debated the ethics of the decision extensively.")
    print("FIXED:", fixed_chunk(sample, "doc1", size=8, overlap=2))
    print("SENTENCE:", sentence_chunk(sample, "doc1", target_size=10))
