import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional

def ingest_document(file_path: str) -> dict:
    """
    Ingest a document into Pinecone vector store.
    
    Args:
        file_path (str): Path to the document to be ingested
    
    Returns:
        dict: Status of the ingestion
    """
    try:
        load_dotenv()
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME must be set in your .env file.")
        
        embedding = OpenAIEmbeddings()
        loader = TextLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        
        PineconeVectorStore.from_documents(
            docs,
            index_name=index_name,
            embedding=embedding,
            namespace="default"
        )
        
        return {
            "status": "success",
            "message": f"Document '{file_path}' successfully ingested into index '{index_name}'",
            "index_name": index_name
        }
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
