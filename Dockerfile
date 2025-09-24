# ------------------------
# Stage 1: Build wheels
# ------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /wheels

# Build deps for native packages (Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and prebuild wheels for hermetic install later
COPY requirements.txt .
RUN python -m pip install -U pip setuptools wheel \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ------------------------
# Stage 2: Runtime image
# ------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    # Sensible defaults for Streamlit if UI is launched
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Create non-root user and working dir
RUN useradd -m -U -s /usr/sbin/nologin appuser
WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser streamlit_app ./streamlit_app
# Runtime-only libs (no headers) for Pillow & TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g libpng16-16 ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps from wheels built in previous stage
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
 && rm -rf /wheels

# Copy application code (API + UI) with correct ownership
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser streamlit_app ./streamlit_app

ENV PYTHONPATH=/app

USER appuser

# Expose both typical ports (API 8000, UI 8501)
EXPOSE 8000 8501

# NOTE: Healthcheck is API-specific; for the Streamlit UI pod, disable it in the Deployment
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz').read()" || exit 1

# Default command runs the API; the UI Deployment overrides this to Streamlit.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]