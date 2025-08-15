from pydantic import BaseModel
import os

class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "deepfake-microservice")
    PORT: int = int(os.getenv("PORT", 8000))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MODEL_PATH: str | None = os.getenv("MODEL_PATH")  # optional for MVP

settings = Settings()