# VerdictMed AI CDSS - Production Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 cdss && \
    useradd --uid 1000 --gid cdss --shell /bin/bash --create-home cdss

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=cdss:cdss src/ ./src/
COPY --chown=cdss:cdss scripts/ ./scripts/
COPY --chown=cdss:cdss configs/ ./configs/

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/models && \
    chown -R cdss:cdss /app/data

# Switch to non-root user
USER cdss

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
