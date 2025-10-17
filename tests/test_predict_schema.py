import base64, io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def _png_bytes():
    # tiny white square
    img = Image.new("RGB", (8, 8), (255, 255, 255))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()

def test_predict_returns_mask_base64():
    r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert "mask_base64" in j, j.keys()
    # must be valid base64 decodable PNG
    raw = base64.b64decode(j["mask_base64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"