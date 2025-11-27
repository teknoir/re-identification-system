# Use a small, secure Python image
FROM python:3.11-slim

# Environment: prevent .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps required by common Python packages and certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user and directories
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install runtime Python dependencies
# Note: Requirements are inferred from the server imports.
RUN pip install --no-cache-dir \
    uvicorn[standard] \
    fastapi \
    requests \
    pymongo \
    google-cloud-storage

# Copy only the manifest editor and its parent package
# If you build from the repo root, this copies the entire project.
COPY . /app/manifest-editor

# Optional defaults for the service; override via `docker run -e ...`
ENV MANIFEST_EDITOR_STATE=/app/gt/manifest_editor_state.json \
    MANIFEST_API_BASE=http://matching-service \
    MANIFEST_EDITOR_MONGO=mongodb://teknoir:change-me@localhost:37017 \
    MANIFEST_EDITOR_DB=gt_tools \
    MANIFEST_EDITOR_ENTRIES_COLL=entries \
    MANIFEST_EDITOR_CLUSTERS_COLL=clusters \
    MANIFEST_EDITOR_GT_COLL=map \
    MANIFEST_EDITOR_BUCKET=gs://victra-poc.teknoir.cloud

# Expose service port
EXPOSE 8000

# Run as non-root
USER appuser

# Start the FastAPI app
# Root path is already set in the application; adjust host/port as needed.
CMD ["uvicorn", "manifest-editor.manifest_editor_server:app", "--host", "0.0.0.0", "--port", "8000"]

