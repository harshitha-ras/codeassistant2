import os
import streamlit as st
import openai
from dotenv import load_dotenv
from unstructured.partition.auto import partition
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
collection = chroma_client.create_collection(name="code_chunks", embedding_function=openai_ef)

def extract_text(file):
    elements = partition(file)
    return "\n".join([str(el) for el in elements])

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def store_chunks(chunks):
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

def semantic_search(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

def generate_response(query, context):
    prompt = f"Context:\n{context}\n\nQuery: {query}\n\nResponse:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

st.title("Coding Assistant with RAG")

uploaded_file = st.file_uploader("Upload a file to add to the database", type=["txt", "py"])
if uploaded_file:
    text = extract_text(uploaded_file)
    chunks = chunk_text(text)
    store_chunks(chunks)
    st.success("File processed and added to the database")

query = st.text_input("Enter your coding question:")

if query:
    relevant_chunks = semantic_search(query)
    context = "\n".join(relevant_chunks)
    response = generate_response(query, context)
    st.write(response)

if __name__ == "__main__":
    st.run()
