# tests/test_infer.py
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient  # type: ignore

from app.main import app

client = TestClient(app)


def _png_bytes(w: int = 32, h: int = 32) -> bytes:
    """
    Generate a tiny deterministic gradient PNG image for tests.
    Shared logic with test_predict, but duplicated here to avoid test import coupling.
    """
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, ((xx + yy) // 2).astype(np.uint8)], axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict():
    """
    Basic end-to-end inference sanity test:
    - /predict returns 200
    - response contains label in {real, fake}
    - score is a normalized float in [0, 1]
    """
    files = {"file": ("sample.png", _png_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200

    data = resp.json()
    assert "label" in data and data["label"] in {"real", "fake"}
    assert "score" in data and 0.0 <= data["score"] <= 1.0