from fastapi import FastAPI, UploadFile, File, HTTPException # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from PIL import Image
from io import BytesIO
from .settings import settings
from .inference import ModelWrapper
from .schemas import HealthResponse, PredictResponse

app = FastAPI(title=settings.APP_NAME)
model = ModelWrapper(model_path=settings.MODEL_PATH)
model.load()

@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model.loaded)

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    content = await file.read()
    try:
        img = Image.open(BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    label, score = model.predict(img)
    return PredictResponse(label=label, score=score)

@app.get("/")
async def root():
    return JSONResponse({"service": settings.APP_NAME, "docs": "/docs"})