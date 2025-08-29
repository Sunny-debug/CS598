from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(json_schema_extra={"example": True})
    model_version: str = Field(json_schema_extra={"example": "v0.1.0"})


class PredictResponse(BaseModel):
    label: str = Field(json_schema_extra={"example": "fake"})
    confidence: float = Field(ge=0, le=1, json_schema_extra={"example": 0.92})
    probs: Dict[str, float] = Field(json_schema_extra={"example": {"real": 0.08, "fake": 0.92}})
    score: float = Field(ge=0, le=1, json_schema_extra={"example": 0.92, "description": "Probability of 'fake'"})
    model_version: str = Field(json_schema_extra={"example": "v0.1.0"})
    threshold: float = Field(ge=0, le=1, json_schema_extra={"example": 0.5})