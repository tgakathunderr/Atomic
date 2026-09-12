import pytest
from fastapi.testclient import TestClient
from demo.api import app, load_model

@pytest.fixture(scope="module")
def client():
    load_model()
    with TestClient(app) as c:
        yield c

def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "atomic-reasoning-prod" in data["model"]
    assert data["parameters"] > 0

def test_api_generate(client):
    payload = {
        "prompt": "Calculate 15 + 25",
        "max_new_tokens": 30,
        "temperature": 0.0
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "completion" in data
    assert "full_text" in data
    assert len(data["completion"]) > 0

def test_api_chat_completions(client):
    payload = {
        "messages": [
            {"role": "user", "content": "What is the capital of logic?"}
        ],
        "max_tokens": 20
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) > 0
    assert "content" in data["choices"][0]["message"]
