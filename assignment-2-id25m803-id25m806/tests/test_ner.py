from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ner_endpoint():
    payload = {"text": "Nitesh Studies at IIT Madras"}
    response = client.post("/api/v1/ner", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) > 0
    labels = [ent["label"] for ent in data["entities"]]
    assert "ORG" in labels
    assert "GPE" in labels  # Have to check whether what is a entity either name nitesh or what

def test_empty_text():
    payload = {"text": ""}
    response = client.post("/api/v1/ner", json=payload)
    assert response.status_code == 400