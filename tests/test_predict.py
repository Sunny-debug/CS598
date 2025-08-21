from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def _png_bytes(color=(128, 128, 128), size=(64, 64)):
    img = Image.new("RGB", size, color=color)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def test_predict_png_works():
    img_bytes = _png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"label", "confidence", "probs", "score", "model_version", "threshold"}
    assert set(data["probs"].keys()) == {"real", "fake"}
    assert 0.0 <= data["score"] <= 1.0

def test_reject_large_image():
    # 11MB+ payload (zeros) to exercise 413
    big = b"\x00" * (11 * 1024 * 1024)
    files = {"file": ("big.png", big, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code in (400, 413)  # may fail as invalid image OR too large