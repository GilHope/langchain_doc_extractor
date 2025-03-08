from langfuse.openai import openai
import os
from dotenv import load_dotenv
from vector import load_vector_store
from text_cleaner import clean_extracted_text
from extractor import extract_text_from_pdf


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def query_vector_store(query, k=10):
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(query, k=k)
    return results

def generate_answer_from_full_doc(query, pdf_path):
    # Extract and clean the full document text
    full_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_extracted_text(full_text)
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    You are an AI assistant analyzing a document.
    
    Based on the context provided, please summarize the key business updates and financial performance details.
    Focus on:
    - Major revenue or operational highlights
    - Key financial performance metrics
    - Strategic initiatives or significant announcements

    Ignore any legal disclaimers or non-GAAP discussions.
    
    **Context:**
    {cleaned_text}
    
    **Question:**
    {query}
    
    **Answer:**
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def generate_answer_from_chunks(query):
    results = query_vector_store(query)
    context_text = "\n".join([doc.page_content for doc in results])
    prompt = f"""
    You are an AI assistant analyzing a document.
    
    Based on the context provided, please summarize the key business updates and financial performance details.
    Focus on:
    - Major revenue or operational highlights
    - Key financial performance metrics
    - Strategic initiatives or significant announcements

    Ignore any legal disclaimers or non-GAAP discussions.
    
    **Context:**
    {context_text}
    
    **Question:**
    {query}
    
    **Answer:**
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    query = "What is the revenue for the quarter?"
    print(generate_answer_from_full_doc(query, "Palantir Q4 2024 Business Update.pdf")) 
