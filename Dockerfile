# ------------------------
# Stage 1: Build wheels
# ------------------------
FROM python:3.11-slim AS builder

WORKDIR /wheels

# Build deps for native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ------------------------
# Stage 2: Runtime image
# ------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user and app dir
RUN useradd -m appuser
WORKDIR /app

# Install deps
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY --chown=appuser:appuser app ./app

# Copy Streamlit UI code
COPY --chown=appuser:appuser streamlit_app ./streamlit_app

ENV PYTHONPATH=/app

# Runtime-only libs (no headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g libpng16-16 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install from prebuilt wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Copy source and ensure ownership
COPY --chown=appuser:appuser app ./app

USER appuser

# Optional: scale via env (set WEB_CONCURRENCY in runtime)
ENV WEB_CONCURRENCY=1

EXPOSE 8000

# Optional: basic healthcheck (expects /healthz route)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]