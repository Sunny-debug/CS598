# app/main.py
from __future__ import annotations

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import HealthResponse, PredictResponse
from app.config import settings
from app.utils.image_io import validate_and_load_image
from app.models.stub import StubDeepfakeModel

logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Deepfake API",
    version=settings.model_version,
    root_path="/api"    # <--- THIS
)

# CORS (tight by default; allow list via env ALLOWED_ORIGINS="http://localhost:8501,http://xyz")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ---- Prometheus metrics ----
# Add middleware BEFORE app starts (module import time)
instrumentator = Instrumentator(
    # You can tweak options here if needed
    # should_instrument_requests_inprogress=True,
)
instrumentator.instrument(app)

@app.on_event("startup")
async def _startup() -> None:
    # Safe to expose the route during startup
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)

# Lazy-load placeholder model (swap later with EfficientNet/ViT)
model = StubDeepfakeModel()

@app.get("/healthz", response_model=HealthResponse)
def healthz():
    return HealthResponse(
        status="ok",
        model_loaded=model.is_loaded(),
        model_version=settings.model_version,
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Basic content-type & size checks are inside validate_and_load_image
    try:
        img = await validate_and_load_image(
            file=file,
            max_mb=settings.max_image_size_mb,
            allowed_content_types={"image/jpeg", "image/png", "image/webp", "image/bmp"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to read/validate image")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    probs = model.predict_proba(img)
    score_fake = float(probs.get("fake", 0.0))
    label = "fake" if score_fake >= settings.threshold else "real"
    confidence = score_fake if label == "fake" else float(probs.get("real", 0.0))

    resp = PredictResponse(
        label=label,
        confidence=confidence,
        probs={"real": float(probs["real"]), "fake": float(probs["fake"])},
        score=score_fake,
        model_version=settings.model_version,
        threshold=settings.threshold,
    )
    logger.info({"event": "predict", "label": resp.label, "score_fake": resp.score, "threshold": settings.threshold})
    return resp