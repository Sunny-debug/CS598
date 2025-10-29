from __future__ import annotations
import os
import base64
import io
import logging
from typing import Dict
from app.config import settings

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError, ImageFile
from prometheus_fastapi_instrumentator import Instrumentator

from app.models.unet_infer import UNetInfer

# Allow loading slightly truncated JPEGs instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Config ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", settings.max_upload_mb)) * 1024 * 1024
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/unet_small_traced.pt")

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Deepfake Detection Service")

# If you host Streamlit on another origin, allow it here:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# Load TorchScript model at startup
try:
    model = UNetInfer(MODEL_PATH)
except FileNotFoundError as e:
    # Fail fast with clear message if model is missing
    logger.error(f"Model load failed: {e}")
    raise

@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness endpoint."""
    return {"status": "ok", "model": MODEL_PATH}

@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """K8s-style health endpoint alias."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Response:
    """
    Accepts an image (jpeg/png), resizes to 256x256 to match training,
    runs UNet, and returns a base64-encoded PNG mask.

    Returns:
        {
          "mask_base64": "<...>",           # grayscale 0-255
          "mask_bin_base64": "<...>",       # optional: binary preview
          "shape": [256, 256]
        }
    """
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    # Enforce an upload size cap (works with most ASGI servers)
    # Note: for absolute enforcement, put a reverse proxy (nginx/traefik) in front too.
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes)")

    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # lightweight structural check
        img = Image.open(io.BytesIO(raw)).convert("RGB")  # re-open after verify
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Bad image upload: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        probs = model.predict(img)  # (256, 256), float32 in [0,1]
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference failed")

    # grayscale 0-255
    mask_img = (np.clip(probs, 0.0, 1.0) * 255).astype(np.uint8)
    # quick binary preview (client can ignore if not needed)
    mask_bin = (mask_img > 128).astype(np.uint8) * 255

    # encode PNGs as base64
    buf = io.BytesIO()
    Image.fromarray(mask_img).save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    buf2 = io.BytesIO()
    Image.fromarray(mask_bin).save(buf2, format="PNG")
    mask_bin_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    return JSONResponse(
        {"mask_base64": mask_b64, "mask_bin_base64": mask_bin_b64, "shape": list(mask_img.shape)}
    )