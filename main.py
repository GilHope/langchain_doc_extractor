from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
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
    vectorstore = FAISS.from_texts(chunks, embeddings)

    faiss_path = "faiss_index"
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path) # delete the existing index
    vectorstore.save_local(faiss_path)

    return vectorstore


def load_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.load_local("faiss_index", embeddings)


def query_vector_store(query, k=3):
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(query, k=k)
    return results



if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(PDF_PATH)
    
    text_chunks = chunk_text(extracted_text)

    vector_store = create_vector_store(text_chunks)

    print(f"Stored {len(text_chunks)} text chunks in FAISS.")

    # Check if FAISS directory exists
    if os.path.exists("faiss_index"):
        print("✅ FAISS index saved successfully.")
    else:
        print("❌ FAISS index was NOT saved. Debug needed.")


