# app/main.py
from __future__ import annotations
import os
import base64
import io
import logging
import time
from typing import Dict, Any

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
MODEL_PATH = os.environ.get("MODEL_PATH", "checkpoints/unet_small_best.ts.pt")
# Threshold used for simple "real/fake" decision
PRED_THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.2"))

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Deepfake Detection Service")

# If you host Streamlit on another origin, allow it here:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in prod to your UI origin
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# --- Prometheus metrics: expose /metrics and add custom metrics ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

from prometheus_client import Summary, Gauge

INFERENCE_LATENCY = Summary(
    "inference_latency_seconds",
    "Time spent on model inference",
)
MASK_COVERAGE = Gauge(
    "mask_coverage_ratio",
    "Mean mask coverage ratio per image",
)

# --- Model state on app ---
app.state.model_loaded = False
app.state.model_version = os.getenv("MODEL_VERSION", os.path.basename(MODEL_PATH))
app.state.infer = None

# Load TorchScript model at startup
try:
    app.state.infer = UNetInfer(MODEL_PATH)
    app.state.model_loaded = True
    logger.info(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError as e:
    # Fail fast with clear message if model is missing
    logger.error(f"Model load failed: {e}")
    # keep model_loaded = False so /healthz reflects reality
    raise


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness endpoint."""
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """K8s-style health endpoint alias."""
    return {
        "status": "ok",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
        "model_version": getattr(app.state, "model_version", "unknown"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Response:
    """
    Accepts an image (jpeg/png), resizes to 256x256 to match training,
    runs UNet, and returns both segmentation masks and a simple
    real/fake classification with scores.

    Returns (example shape):

        {
          "label": "real" | "fake",
          "confidence": <float>,
          "probs": {"real": <float>, "fake": <float>},
          "score": <float>,
          "model_version": "<string>",
          "threshold": <float>,
          "mask_base64": "<...>",
          "mask_bin_base64": "<...>",
          "shape": [256, 256],
          "metrics": {
             "inference_latency_s": <float>,
             "mask_coverage_ratio": <float>
          }
        }
    """
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    # Enforce an upload size cap (works with most ASGI servers)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes)",
        )

    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()  # lightweight structural check
        img = Image.open(io.BytesIO(raw)).convert("RGB")  # re-open after verify
    except (UnidentifiedImageError, OSError) as e:
        logger.warning(f"Bad image upload: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Ensure model is available
    infer = getattr(app.state, "infer", None)
    if infer is None:
        logger.error("Inference called but model is not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    # --- Inference with metrics ---
    try:
        t0 = time.perf_counter()
        probs = infer.predict(img)  # (H, W), float32 in [0,1]
        latency = time.perf_counter() - t0
        INFERENCE_LATENCY.observe(latency)  # record latency
    except Exception as e:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference failed")

    # grayscale 0-255
    probs_clipped = np.clip(probs, 0.0, 1.0)
    mask_img = (probs_clipped * 255).astype(np.uint8)
    # quick binary preview (client can ignore if not needed)
    mask_bin = (mask_img > 128).astype(np.uint8) * 255

    # coverage metric from the probabilities
    coverage = float(probs_clipped.mean())
    MASK_COVERAGE.set(coverage)  # record coverage ratio

    # --- Simple classification on top of coverage ---
    # Treat coverage as a "fake score" in [0,1]
    score = coverage
    score = float(max(0.0, min(1.0, score)))

    # label based on threshold
    label = "fake" if score >= PRED_THRESHOLD else "real"

    probs_dict = {
        "real": float(1.0 - score),
        "fake": float(score),
    }
    confidence = float(max(probs_dict.values()))

    # encode PNGs as base64
    buf = io.BytesIO()
    Image.fromarray(mask_img).save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    buf2 = io.BytesIO()
    Image.fromarray(mask_bin).save(buf2, format="PNG")
    mask_bin_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    return JSONResponse(
        {
            # classification contract
            "label": label,
            "confidence": confidence,
            "probs": probs_dict,
            "score": score,
            "model_version": getattr(app.state, "model_version", "unknown"),
            "threshold": PRED_THRESHOLD,
            # segmentation + metrics
            "mask_base64": mask_b64,
            "mask_bin_base64": mask_bin_b64,
            "shape": list(mask_img.shape),
            "metrics": {
                "inference_latency_s": latency,
                "mask_coverage_ratio": coverage,
            },
        }
    )
