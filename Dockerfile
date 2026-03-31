FROM python:3.11-slim-bullseye

# System deps for scientific Python (nilearn, matplotlib, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps before copying source (better layer caching)
COPY requirements.txt pyproject.toml ./
COPY src/notion_health_ai/__init__.py src/notion_health_ai/__init__.py
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e "." && \
    pip install --no-cache-dir uvicorn[standard]

# Copy full source
COPY . .

# Create dirs that need to exist at runtime
RUN mkdir -p visualizations logs

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check hits the /api/status endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["uvicorn", "notion_health_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
