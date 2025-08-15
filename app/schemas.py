from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(example="ok")
    model_loaded: bool = Field(example=True)

class PredictResponse(BaseModel):
    label: str = Field(example="fake")
    score: float = Field(ge=0.0, le=1.0, example=0.83)