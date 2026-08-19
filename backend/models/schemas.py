"""
Typed contracts for every stage of process_query(). If a stage's
output doesn't match these shapes, it fails loudly at the boundary
instead of silently corrupting the next stage.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"
    METADATA_AWARE = "metadata_aware"


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_id: Optional[str] = None
    language: str
    strategy: ChunkStrategy
    text: str
    query_type: Optional[str] = None
    is_selected_gt: Optional[int] = None  # ground-truth relevance flag, eval only


class RetrievedChunk(BaseModel):
    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0


class RelevanceGateResult(BaseModel):
    passed: bool
    top_fused_score: float
    reason: str


class Citation(BaseModel):
    chunk_id: str
    reason: str


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    model_confidence: Optional[float] = None  # display only, NEVER gates decisions


class GroundingResult(BaseModel):
    supported: bool
    score: float
    unsupported_claims: list[str] = Field(default_factory=list)


class GuardrailFlag(str, Enum):
    NONE = "none"
    OFF_TOPIC = "off_topic"
    UNSAFE_INPUT = "unsafe_input"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    GROUNDING_FAILURE = "grounding_failure"
    INVALID_INPUT = "invalid_input"
    SERVICE_FAILURE = "service_failure"


class StageLatency(BaseModel):
    stt_ms: Optional[float] = None
    query_processing_ms: Optional[float] = None
    vector_search_ms: Optional[float] = None
    bm25_search_ms: Optional[float] = None
    fusion_ms: Optional[float] = None
    relevance_gate_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    grounding_ms: Optional[float] = None
    total_ms: Optional[float] = None
    # Two totals reported separately — STT latency is network/API bound
    # and outside the <200ms retrieval+generation budget by design.
    post_transcript_ms: Optional[float] = None


class QueryRequest(BaseModel):
    text: Optional[str] = None          # typed fallback
    audio_base64: Optional[str] = None  # voice input
    language_hint: Optional[str] = None


class QueryResponse(BaseModel):
    status: Literal["answered", "abstained", "blocked", "error"]
    transcript: Optional[str] = None
    answer: Optional[GeneratedAnswer] = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    grounding: Optional[GroundingResult] = None
    guardrail_flag: GuardrailFlag = GuardrailFlag.NONE
    message: Optional[str] = None
    latency: StageLatency = Field(default_factory=StageLatency)
