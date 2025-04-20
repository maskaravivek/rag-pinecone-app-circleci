# tests/test_api.py
import json
from api.server import app

def test_ask_endpoint():
    client = app.test_client()
    response = client.post('/ask', json={"question": "What is Pinecone?"})
    data = response.get_json()

    assert response.status_code == 200
    assert "answer" in data
    assert data["answer"]
