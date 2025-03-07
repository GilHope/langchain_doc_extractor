from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import pdfplumber
import os 
import shutil
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PDF_PATH = "Palantir Q4 2024 Business Update.pdf"

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    
    return text


def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma.from_texts(chunks, embeddings, persist_directory="chroma_db")# Save to disk
    return vectorstore


def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)


def query_vector_store(query, k=3):
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(query, k=k)
    return results



if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(PDF_PATH)
    
    text_chunks = chunk_text(extracted_text)

    vector_store = create_vector_store(text_chunks)
    # Query Example
    query = "What is Palantir's financial outlook?"
    results = query_vector_store(query)

    print("\n🔍 Query Results:")
    for i, result in enumerate(results):
        print(f"\nChunk {i+1}:")
        print(result.page_content)


