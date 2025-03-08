import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    return text

if __name__ == "__main__":
    print("Extracting text from PDF...")
    text = extract_text_from_pdf("Palantir Q4 2024 Business Update.pdf")
    print(text)
    with open("extracted_text.txt", "w") as f:
        f.write(text)