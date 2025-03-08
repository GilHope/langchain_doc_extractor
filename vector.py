from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chroma_db"

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    documents = [Document(page_content=chunk, metadata={"chunk_index": i}) for i, chunk in enumerate(chunks)]
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_PATH)
    print(f"✅ Stored {len(chunks)} text chunks in ChromaDB.")
    return vectorstore

def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
