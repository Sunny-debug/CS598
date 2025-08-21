from pydantic import BaseModel
import os

class Settings(BaseModel):
    app_name: str = "deepfake-microservice"
    model_version: str = os.getenv("MODEL_VERSION", "v0.1.0")
    threshold: float = float(os.getenv("THRESHOLD", "0.5"))
    max_image_size_mb: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))

settings = Settings()