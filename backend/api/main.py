import sys
import time
import base64
import logging
import traceback
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "services"))

from config import settings  # noqa: E402
from services.harness import process_query, process_query_fast  # noqa: E402
from services.stt import transcribe_audio  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-rag-api")

app = FastAPI(title="Voice RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def json_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s", request.method, request.url.path)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "detail": "See server logs for the full traceback.",
        },
    )


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


class QueryRequest(BaseModel):
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    audio_content_type: Optional[str] = "audio/webm"
    mode: str = "fast"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query")
def query(req: QueryRequest):
    t_start = time.perf_counter()
    stt_ms = None
    transcript = req.text

    if req.audio_base64:
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64 audio: {e}")

        stt_result = transcribe_audio(audio_bytes, content_type=req.audio_content_type or "audio/webm")
        stt_ms = stt_result["elapsed_ms"]

        if stt_result["error"]:
            return {
                "status": "error",
                "transcript": None,
                "message": f"Speech-to-text failed: {stt_result['error']}",
                "latency": {"stt_ms": stt_ms, "total_ms": (time.perf_counter() - t_start) * 1000},
            }

        transcript = stt_result["text"]
        if not transcript:
            return {
                "status": "blocked",
                "transcript": "",
                "guardrail_flag": "invalid_input",
                "message": "Could not transcribe any speech from the audio.",
                "latency": {"stt_ms": stt_ms, "total_ms": (time.perf_counter() - t_start) * 1000},
            }

    if req.text is None and not req.audio_base64:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'audio_base64'.")
    if transcript is None:
        transcript = ""

    if req.mode == "polished":
        result = process_query(transcript)
    else:
        result = process_query_fast(transcript)

    result["transcript"] = transcript
    if stt_ms is not None:
        result["latency"]["stt_ms"] = stt_ms
        result["latency"]["total_ms"] = stt_ms + result["latency"].get(
            "end_to_end_ms", result["latency"]["fast_path_total_ms"]
        )

    return _json_safe(result)


FRONTEND_DIR = REPO_ROOT / "frontend"
if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    logger.warning("frontend/index.html not found - serving API only, no UI mounted at /")
