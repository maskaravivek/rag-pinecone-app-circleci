# app/query.py
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

def ask(question: str) -> str:
    embedding = OpenAIEmbeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME must be set in your .env file.")
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
        namespace="default"
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        retriever=docsearch.as_retriever()
    )
    
    result = qa.invoke({"query": question})
    return result["result"]
