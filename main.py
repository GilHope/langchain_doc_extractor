from extractor import extract_text_from_pdf
from chunker import chunk_text
from vector import create_vector_store
from query_handler import generate_answer_from_chunks

PDF_PATH = "Palantir Q4 2024 Business Update.pdf"

def main():
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(PDF_PATH)
    
    # Split text into chunks
    text_chunks = chunk_text(extracted_text)
    
    # Create vector store
    create_vector_store(text_chunks)
    
    # Run
    query = "What is this document about?"
    answer = generate_answer_from_chunks(query)
    
    print("\n🔍 AI Response:")
    print(answer)

if __name__ == "__main__":
    main()
