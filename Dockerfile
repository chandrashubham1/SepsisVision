# SepsisAI BEI API — Dockerfile
# Adnexus Biotechnologies Inc | INVESTIGATIONAL — Not FDA Cleared
#
# Build:
#   docker build -t sepsisai-api .
#
# Run (no ChromaDB volume):
#   docker run -p 8000:8000 --env-file .env sepsisai-api
#
# Run (with persistent ChromaDB and patient cache):
#   docker run -p 8000:8000 --env-file .env \
#     -v /path/to/chroma_db:/data/chroma_db \
#     -v /path/to/patient_cache:/data/patient_cache \
#     sepsisai-api

FROM python:3.11-slim

# --- System deps for PyMuPDF and sentence-transformers ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Install Python dependencies (layer-cached unless requirements change) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Pre-download sentence-transformer models at build time ---
# Avoids cold-start delays when the container first handles a request.
# Remove this block if you prefer lazy loading (smaller image, slower first request).
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('NeuML/pubmedbert-base-embeddings'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

# --- Copy application ---
COPY *.py ./
COPY tool.html landing.html ./
COPY static/ ./static/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# --- Data directories (overridden by volume mounts at runtime) ---
RUN mkdir -p /data/chroma_db /data/patient_cache

# --- Non-root user ---
RUN addgroup --system sepsisai && adduser --system --ingroup sepsisai sepsisai
RUN chown -R sepsisai:sepsisai /app /data
USER sepsisai

# --- Expose port ---
EXPOSE 8000

# --- Healthcheck ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Start server ---
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
