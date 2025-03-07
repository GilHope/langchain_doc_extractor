import pdfplumber

PDF_PATH = "Palantir Q4 2024 Business Update.pdf"

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    
    return text




if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(PDF_PATH)
    print(f"Extracted {len(extracted_text.split())} words from the PDF.\n")
    print(extracted_text[:1000]) 