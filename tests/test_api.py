import os
from fastapi.testclient import TestClient
from app.main import app

os.environ["GROQ_API_KEY"] = "test-key"
os.environ["MISTRAL_API_KEY"] = "test-key"
os.environ["NVIDIA_API_KEY"] = "test-key"

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "groq" in data["providers"]
    assert data["models"] > 0

def test_serve_frontend():
    r = client.get("/")
    assert r.status_code == 200

def test_unknown_model():
    r = client.post("/v1/chat/completions", json={"model": "nonexistent"})
    assert r.status_code == 400
