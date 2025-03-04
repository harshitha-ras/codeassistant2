import streamlit as st
import openai
import pinecone
from datasets import load_dataset
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index_name = "code-assistant"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # OpenAI embeddings are 1536 dimensions
index = pinecone.Index(index_name)

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

def load_and_process_data():
    dataset = load_dataset("code_search_net", "python", split="train", streaming=True)
    for i, item in enumerate(dataset.take(1000)):  # Limit to 1000 examples for demonstration
        code = item['code']
        embedding = get_embedding(code)
        index.upsert([(str(i), embedding, {"code": code})])

def semantic_search(query, k=3):
    query_embedding = get_embedding(query)
    results = index.query(query_embedding, top_k=k, include_metadata=True)
    return [match['metadata']['code'] for match in results['matches']]

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

st.title("Coding Assistant with RAG (Pinecone)")

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
