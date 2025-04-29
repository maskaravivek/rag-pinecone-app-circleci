# tests/test_ingest.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import patch, MagicMock
from app.ingest import ingest_document

@patch("app.ingest.PineconeVectorStore")
@patch("app.ingest.OpenAIEmbeddings")
@patch("app.ingest.TextLoader")
@patch("app.ingest.RecursiveCharacterTextSplitter")
def test_ingest_document_success(mock_text_splitter, mock_text_loader, mock_embeddings, mock_vectorstore):
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = ["mock_document"]
    mock_text_loader.return_value = mock_loader_instance

    mock_splitter_instance = MagicMock()
    mock_splitter_instance.split_documents.return_value = ["mock_chunk1", "mock_chunk2"]
    mock_text_splitter.return_value = mock_splitter_instance

    mock_embeddings.return_value = MagicMock()
    mock_vectorstore.from_documents.return_value = None

    with patch.dict("os.environ", {"PINECONE_INDEX_NAME": "mock-index"}):
        result = ingest_document("mock_file.txt")

    assert result["status"] == "success"
    assert result["index_name"] == "mock-index"
    assert "successfully ingested" in result["message"]
    
    mock_text_loader.assert_called_once_with("mock_file.txt")
    mock_loader_instance.load.assert_called_once()
    mock_text_splitter.assert_called_once()
    mock_splitter_instance.split_documents.assert_called_once_with(["mock_document"])
    mock_embeddings.assert_called_once()
    mock_vectorstore.from_documents.assert_called_once_with(
        ["mock_chunk1", "mock_chunk2"],
        index_name="mock-index",
        embedding=mock_embeddings.return_value,
        namespace="default"
    )

def test_ingest_document_missing_env():
    with patch.dict("os.environ", {}, clear=True):
        result = ingest_document("mock_file.txt")
    
    assert result["status"] == "error"
    assert "PINECONE_INDEX_NAME must be set" in result["message"]
