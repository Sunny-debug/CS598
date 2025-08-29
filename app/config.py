from __future__ import annotations

import os
from typing import List
from pydantic import BaseModel


def _parse_csv_list(val: str | None) -> List[str]:
    if not val:
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


class Settings(BaseModel):
    # App
    app_name: str = os.getenv("APP_NAME", "deepfake-microservice")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model_version: str = os.getenv("MODEL_VERSION", "v0.1.0")

    # Model behavior
    model_path: str | None = os.getenv("MODEL_PATH")  # optional (MVP uses stub)
    threshold: float = float(os.getenv("THRESHOLD", "0.5"))

    # Uploads
    max_image_size_mb: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))

    # CORS
    allowed_origins: List[str] = _parse_csv_list(os.getenv("ALLOWED_ORIGINS", ""))


settings = Settings()