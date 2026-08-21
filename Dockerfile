# Single-container deploy: FastAPI serves both /api/* and the
# frontend (see backend/api/main.py's StaticFiles mount). Prebuilt
# FAISS/BM25 indexes ship inside the image (data/indexes/) so there's
# no ingestion step at container startup.
FROM python:3.11-slim

WORKDIR /app

# Install basic build tools required for faiss compilation steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

# Install lightweight ONNX and transformers dependencies from requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
# ONNX model files already exist on disk (backend/models/e5-small-onnx) - no download/pre-cache step needed, they get baked in via the COPY below.




# Copy all project directories into the image container.
# Your pre-downloaded model assets in backend/models/e5-small-onnx 
# are automatically backed into the image during this layer.
COPY backend /app/backend
COPY retrieval /app/retrieval
COPY frontend /app/frontend
COPY data /app/data

# Environment optimizations for hosting environments
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
