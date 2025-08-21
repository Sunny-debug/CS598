from typing import Dict
from PIL import Image
import numpy as np

from .base import DeepfakeModel

class StubDeepfakeModel(DeepfakeModel):
    """
    Drop-in placeholder—no torch/timm.
    Uses a simple heuristic (average brightness + edge magnitude) to produce a
    stable, deterministic 'fake' score. Later, replace with an EfficientNet/ViT implementation.
    """

    def __init__(self):
        self._loaded = True  # Simulate ready state

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_proba(self, img: Image.Image) -> Dict[str, float]:
        arr = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)
        # Simple, deterministic heuristics to mimic a classifier:
        brightness = float(arr.mean())               # 0..1
        contrast = float(arr.std()) * 2.0            # 0..~1
        # Edge-ish proxy via channel differences:
        edges = float(np.mean(np.abs(np.diff(arr, axis=0))))  # 0..1-ish

        # Mix into a pseudo "fake" score (clamped 0..1)
        fake_score = brightness * 0.3 + contrast * 0.5 + edges * 0.4
        fake_score = max(0.0, min(1.0, fake_score))

        real = 1.0 - fake_score
        fake = fake_score
        # Normalize (just in case of rounding)
        s = real + fake
        real, fake = real / s, fake / s
        return {"real": real, "fake": fake}