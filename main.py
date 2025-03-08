from extractor import extract_text_from_pdf
from text_cleaner import clean_extracted_text
from chunker import chunk_text
from vector import create_vector_store
from query_handler import generate_answer_from_chunks

PDF_PATH = "Palantir Q4 2024 Business Update.pdf"

def main():
    # Extract text from PDF
    extracted_text = extract_text_from_pdf(PDF_PATH)

    # Clean the extracted text to remove boilerplate (e.g., repeated headers/footers)
    cleaned_text = clean_extracted_text(extracted_text)

    # # clean inspection
    # cleaned_text = clean_extracted_text(extracted_text)
    # print("🔍 Cleaned Text Preview (first 1000 characters):")
    # print(cleaned_text[:1000])
    # print("\n-------------------------\n")
    
    # Split into chunks
    text_chunks = chunk_text(extracted_text)

    # # chunk inspection
    # text_chunks = chunk_text(cleaned_text)
    # for i, chunk in enumerate(text_chunks):
    #     print(f"Chunk {i} Preview (first 300 characters):")
    #     print(chunk[:300])
    #     print("\n-------------------------\n")
    
    # Create vector store
    create_vector_store(text_chunks)
    
    # Run
    query = "Summarize the key business updates and financial performance details from the document. Please ignore legal disclaimers and non-GAAP discussions."

    answer = generate_answer_from_chunks(query)
    
    print("\n🔍 AI Response:")
    print(answer)

if __name__ == "__main__":
    main()
