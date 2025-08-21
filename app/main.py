from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from .schemas import HealthResponse, PredictResponse
from .config import settings
from .utils.image_io import load_image_from_bytes, downscale_if_needed
from .models.stub import StubDeepfakeModel

import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Deepfake Image Detection Microservice", version="0.1.0")

# CORS (tight by default; update when you attach a UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Lazy-load the placeholder model
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
    # Basic content-type & size checks
    if file.content_type not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {file.content_type}")

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.max_image_size_mb:
        raise HTTPException(status_code=413, detail=f"Image too large (> {settings.max_image_size_mb} MB).")

    try:
        img = load_image_from_bytes(raw)
        img = downscale_if_needed(img, max_side=1024)
    except Exception as e:
        logger.exception("Failed to load image")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    probs = model.predict_proba(img)
    score_fake = float(probs.get("fake", 0.0))
    label = "fake" if score_fake >= settings.threshold else "real"

    # Confidence = probability of predicted label
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

# Optional: plain /metrics placeholder to be replaced by Prometheus client later
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return "# TODO: integrate prometheus_client and export real metrics\n"