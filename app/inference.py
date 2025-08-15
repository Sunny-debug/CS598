from __future__ import annotations
from typing import Optional
from PIL import Image
import numpy as np
import os

class DummyModel:
    """A tiny heuristic model so the service runs without heavy deps.
    
    It computes a simple statistic (edge intensity proxy) and maps to a
    probability. Replace with a real model later.
    """

    def __init__(self) -> None:
        pass

    def predict_proba(self, img: Image.Image) -> float:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
        # simple high-frequency proxy via finite differences
        dx = np.diff(arr, axis=1, prepend=arr[:, :1])
        dy = np.diff(arr, axis=0, prepend=arr[:1, :])
        energy = (np.abs(dx) + np.abs(dy)).mean()
        # normalize into [0,1] using a soft mapping
        score = 1.0 - np.exp(-energy / 40.0)
        # clamp for safety
        return float(np.clip(score, 0.0, 1.0))

class ModelWrapper:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path if model_path and os.path.exists(model_path) else None
        self.model = None

    def load(self) -> None:
        # In MVP, fall back to DummyModel if no valid model_path
        self.model = DummyModel()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, img: Image.Image) -> tuple[str, float]:
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        score = self.model.predict_proba(img)
        label = "fake" if score >= 0.5 else "real"
        return label, score