import openai
import os
from dotenv import load_dotenv
from vector import load_vector_store

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def query_vector_store(query, k=3):
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(query, k=k)
    return results

def generate_answer_from_chunks(query):
    results = query_vector_store(query)
    context_text = "\n".join([doc.page_content for doc in results])
    prompt = f"""
    You are an AI assistant analyzing a document.
    
    Answer the question based on the context provided.
    
    **Context:**
    {context_text}
    
    **Question:**
    {query}
    
    **Answer:**
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
