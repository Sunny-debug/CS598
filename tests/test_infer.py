from fastapi.testclient import TestClient # type: ignore
from PIL import Image
import io
import numpy as np
from app.main import app

client = TestClient(app)

def _png_bytes(w: int = 32, h: int = 32) -> bytes:
    # generate a tiny gradient image deterministically
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, ((xx+yy)//2).astype(np.uint8)], axis=-1)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_predict():
    files = {"file": ("sample.png", _png_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data and data["label"] in {"real", "fake"}
    assert "score" in data and 0.0 <= data["score"] <= 1.0