# tests/test_query.py
from app.query import ask

def test_question():
    result = ask("What is Pinecone?")
    assert result is not None
