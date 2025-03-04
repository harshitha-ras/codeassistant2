import streamlit as st
import openai
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB
chroma_client = chromadb.Client()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
collection = chroma_client.create_collection(name="code_chunks", embedding_function=openai_ef)

def load_and_process_data():
    dataset = load_dataset("code_search_net", "python", split="train", streaming=True)
    for item in dataset.take(1000):  # Limit to 1000 examples for demonstration
        code = item['code']
        collection.add(
            documents=[code],
            ids=[f"chunk_{item['url']}"]
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

if 'data_loaded' not in st.session_state:
    with st.spinner("Loading and processing data..."):
        load_and_process_data()
    st.session_state.data_loaded = True

query = st.text_input("Enter your coding question:")

if query:
    relevant_chunks = semantic_search(query)
    context = "\n".join(relevant_chunks)
    response = generate_response(query, context)
    st.write(response)
