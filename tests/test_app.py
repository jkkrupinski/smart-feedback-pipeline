from fastapi.testclient import TestClient
from docker.fastapi_app.app.main import app


client = TestClient(app)


def test_predict_endpoint():
    sample_input = {"text": "I love this product"}
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "label" in response.json()
