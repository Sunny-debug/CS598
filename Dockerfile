# lightweight, reproducible
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Security: create non-root user
RUN useradd -m appuser
WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install requirements first (better layer cache)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY app ./app

# Switch to non-root
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8000"]