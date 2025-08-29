from __future__ import annotations

from typing import Dict
import numpy as np
from PIL import Image

from app.models.base import DeepfakeModel


class StubDeepfakeModel(DeepfakeModel):
    """
    Drop-in placeholder — no torch/timm.
    Deterministic heuristic based on brightness/contrast/edge-ish proxy.
    """

    def __init__(self):
        self._loaded = True  # Simulate ready state

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_proba(self, img: Image.Image) -> Dict[str, float]:
        arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)
        brightness = float(arr.mean())               # 0..1
        contrast = float(arr.std()) * 2.0            # 0..~1
        edges = float(np.mean(np.abs(np.diff(arr, axis=0))))  # 0..1-ish

        fake_score = brightness * 0.3 + contrast * 0.5 + edges * 0.4
        fake_score = max(0.0, min(1.0, fake_score))

        real = 1.0 - fake_score
        fake = fake_score
        s = real + fake
        return {"real": real / s, "fake": fake / s}