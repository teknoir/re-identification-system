FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps kept minimal; add/remove as needed for your matcher (Torch, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy source
COPY . /app/matching-service

# Install Python deps (replace / extend as needed; prefer requirements.txt if available)
# If a requirements.txt exists you can replace this with: COPY requirements.txt . && pip install -r requirements.txt
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pymongo \
    numpy \
    scikit-learn \
    torch \
    faiss-cpu

# Optional: uncomment to install torch if required (large layer)
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Runtime environment defaults (override with -e)
ENV MODEL_CKPT="matching-service/models/encoder/model.pt" \
    ATTR_SCHEMA="attr_schema.json" \
    BUCKET_PREFIX="gs://victra-poc.teknoir.cloud" \
    REID_MONGODB_URI="mongodb://teknoir:change-me@localhost:37017" \
    MONGO_DB="reid_service" \
    MONGO_ENTRIES_COLLECTION="observations" \
    MONGO_EVENTS_COLLECTION="line-crossings" \
    THRESHOLD="0.753" \
    TOPK="20"

EXPOSE 8000

# Healthcheck to fail container if service unresponsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

USER appuser

# Start FastAPI via uvicorn
CMD ["uvicorn", "matching-service.app:app", "--host", "0.0.0.0", "--port", "8000"]

