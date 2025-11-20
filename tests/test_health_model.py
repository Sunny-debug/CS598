# tests/test_health_model.py
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_includes_model():
    # We hit the in-process app, not a real HTTP server
    r = client.get("/health")
    assert r.status_code == 200

    data = r.json()
    # Original intent: ensure model info is present
    assert "model" in data
    assert isinstance(data["model"], str)

    # And optionally also validate the richer /healthz endpoint
    r2 = client.get("/healthz")
    assert r2.status_code == 200
    h2 = r2.json()
    assert "model_loaded" in h2
    assert isinstance(h2["model_loaded"], bool)