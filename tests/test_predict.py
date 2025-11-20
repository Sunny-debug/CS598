# tests/test_predict.py
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient  # type: ignore

from app.main import app

client = TestClient(app)


def _png_bytes(w: int = 32, h: int = 32) -> bytes:
    """
    Generate a tiny deterministic gradient PNG image for tests.
    """
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, ((xx + yy) // 2).astype(np.uint8)], axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_png_works():
    img_bytes = _png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()

    required = {"label", "confidence", "probs", "score", "model_version", "threshold"}
    # Only require these keys to be present; extra keys (mask_base64, metrics, etc.) are allowed
    assert required.issubset(set(data.keys()))

    assert data["label"] in {"real", "fake"}
    assert isinstance(data["confidence"], float)
    assert isinstance(data["score"], float)
    assert isinstance(data["threshold"], float)