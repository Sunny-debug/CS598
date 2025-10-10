FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_CONCURRENCY=1

# OS libs for Pillow & TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo libpng16-16 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install PyTorch CPU wheels explicitly (small + reliable) ---
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# --- Install your app requirements (NO torch/torchvision here) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code only (model will be volume-mounted at runtime)
COPY app ./app

# (optional) add both health endpoints; or keep only /health in your app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]