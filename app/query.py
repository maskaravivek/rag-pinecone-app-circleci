# app/query.py
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
embedding = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), embedding)

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=docsearch.as_retriever())

def ask(question):
    return qa.run(question)
