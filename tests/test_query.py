# tests/test_query.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from app.query import ask

@patch("app.query.PineconeVectorStore")
@patch("app.query.OpenAIEmbeddings")
@patch("app.query.ChatOpenAI")
@patch("app.query.RetrievalQA")
def test_ask_success(mock_retrieval_qa, mock_chat_openai, mock_embeddings, mock_vectorstore):
    mock_docsearch = MagicMock()
    mock_vectorstore.from_existing_index.return_value = mock_docsearch

    mock_llm = MagicMock()
    mock_chat_openai.return_value = mock_llm

    mock_qa_chain = MagicMock()
    mock_qa_chain.invoke.return_value = {"result": "mocked answer"}
    mock_retrieval_qa.from_chain_type.return_value = mock_qa_chain

    with patch.dict("os.environ", {"PINECONE_INDEX_NAME": "mock-index"}):
        response = ask("What is RAG?")
    
    assert response == "mocked answer"

    mock_vectorstore.from_existing_index.assert_called_once_with(
        index_name="mock-index",
        embedding=mock_embeddings.return_value,
        namespace="default"
    )
    mock_chat_openai.assert_called_once()
    mock_retrieval_qa.from_chain_type.assert_called_once_with(
        llm=mock_chat_openai.return_value,
        retriever=mock_docsearch.as_retriever.return_value
    )
    mock_qa_chain.invoke.assert_called_once_with({"query": "What is RAG?"})

@patch("app.query.OpenAIEmbeddings")
def test_ask_missing_env(mock_embeddings):
    mock_embeddings.return_value = None

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            ask("Test question")

    assert "PINECONE_INDEX_NAME must be set" in str(excinfo.value)
