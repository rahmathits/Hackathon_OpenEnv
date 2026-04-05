# ─────────────────────────────────────────────────────────────
# EDA OpenEnv — Dockerfile
# Uses pip instead of uv for reliable Docker builds
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="EDA OpenEnv"
LABEL description="Real-world RL environment for EDA pipeline agents"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app:${PATH}"
ENV PYTHONPATH="/app"
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME=""
ENV ENABLE_WEB_INTERFACE = true

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project source
COPY models.py          .
COPY pipeline.py        .
COPY grader.py          .
COPY openenv.yaml       .
COPY client.py          .
COPY inference.py       .
COPY server/            ./server/

# Create init files
RUN touch __init__.py && touch server/__init__.py

RUN mkdir -p /app/data

# Healthcheck — validator pings /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]