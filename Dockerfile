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
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
# Pre-download the Hugging Face embedding model during build time
# Pre-download the SentenceTransformer model during build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"


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
