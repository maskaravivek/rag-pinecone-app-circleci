# app/ingest.py
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
import pinecone

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

loader = TextLoader("sample_docs.txt")
docs = loader.load()

embedding = OpenAIEmbeddings()

index_name = os.getenv("PINECONE_INDEX_NAME")
Pinecone.from_documents(docs, embedding, index_name=index_name)
