# Single-container deploy: FastAPI serves both /api/* and the
# frontend (see backend/api/main.py's StaticFiles mount). Prebuilt
# FAISS/BM25 indexes ship inside the image (data/indexes/) so there's
# no ingestion step at container startup - only the embedding model
# downloads on first request (see _lazy_load() in retrieval/hybrid_search.py).
FROM python:3.11-slim

WORKDIR /app

# faiss-cpu and sentence-transformers/torch pull in some packages that
# want basic build tools even when installing prebuilt wheels on
# certain platforms - keep this minimal but present.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
# CPU-only torch wheel installed first and pinned - without this,
# sentence-transformers pulls the default CUDA build (800MB+ of
# unused NVIDIA libs), which is what OOM'd the 512MB Render tier.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY retrieval /app/retrieval
COPY frontend /app/frontend
COPY data /app/data

# HF Spaces' Docker SDK expects the app on port 7860. Falls back to
# $PORT (Render/Railway/Fly convention) if set, so this same image
# works on any of them without edits.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
