from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import pdfplumber
import os 
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




if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(PDF_PATH)
    
    text_chunks = chunk_text(extracted_text)

    print(f"Total Chunks Created: {len(text_chunks)}")
    print("\nSample Chunk:\n", text_chunks[0])

