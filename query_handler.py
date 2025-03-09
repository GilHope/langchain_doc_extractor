from langfuse.openai import openai
import os
from dotenv import load_dotenv
from vector import load_vector_store
from text_cleaner import clean_extracted_text
from extractor import extract_text_from_pdf



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_API_SECRET_KEY = os.getenv("LANGFUSE_API_SECRET_KEY")
LANGFUSE_API_PUBLIC_KEY = os.getenv("LANGFUSE_API_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")


def rewrite_query(original_query):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""Rewrite the following query to be more effective for vector store retrieval. Follow these rules:
    1. Focus on key terms and concepts that would appear in the text
    2. Remove conversational elements
    3. Include relevant synonyms or alternative phrasings in parentheses
    4. Keep it concise but information-dense
    5. Use specific business/financial terminology where appropriate
    
    For example:
    User query: "How did the company do last quarter?"
    Rewritten: "quarterly financial performance revenue (earnings) profit margins growth metrics"
    
    User query: "What are they planning to do about AI?"
    Rewritten: "artificial intelligence (AI) strategy initiatives development roadmap investments"
    
    Original query: {original_query}
    
    Rewritten query:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant that specializes in optimizing queries for vector store retrieval of business documents."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def query_vector_store(query, k=10):
    # First rewrite the query for better semantic search
    rewritten_query = rewrite_query(query)
    print(f"Original query: {query}")
    print(f"Rewritten query: {rewritten_query}")  # Added for debugging
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(rewritten_query, k=k)
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
