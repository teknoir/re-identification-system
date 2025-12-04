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

# Copy requirements
COPY requirements.txt /app/manifest-editor/requirements.txt
# Install Python deps (replace / extend as needed; prefer requirements.txt if available)
RUN pip install --upgrade pip && pip install --no-cache-dir -r ./manifest-editor/requirements.txt

# Copy source
COPY . /app/manifest-editor

# Expose service port
EXPOSE 8000

# Run as non-root
USER appuser

# Start the FastAPI app
# Root path is already set in the application; adjust host/port as needed.
CMD ["uvicorn", "manifest-editor.manifest_editor_server:app", "--host", "0.0.0.0", "--port", "8000"]

