import base64, json
from httpx import AsyncClient
from app.main import app
from PIL import Image
import io, numpy as np
import pytest

@pytest.mark.asyncio
async def test_predict_returns_mask_base64():
    # create a tiny dummy image in-memory
    arr = (np.random.rand(64,64,3)*255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG"); buf.seek(0)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        files = {"file": ("x.jpg", buf.getvalue(), "image/jpeg")}
        r = await ac.post("/predict", files=files)
    assert r.status_code == 200
    j = r.json()
    assert "mask_base64" in j and isinstance(j["mask_base64"], str) and len(j["mask_base64"]) > 0