# tests/test_health_model.py
import os, requests

def test_health_includes_model():
    api = os.getenv("API_BASE", "http://127.0.0.1:8000")
    h = requests.get(f"{api}/health", timeout=5).json()
    assert "model" in h, "health must expose active model"
    # should include weights when UNet is active
    if (h.get("model") or "").startswith("unet"):
        assert h.get("weights_sha256"), "weights SHA should be present for UNet"