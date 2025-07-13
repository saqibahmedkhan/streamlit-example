import os
import faiss
import numpy as np
from typing import List
from openai import OpenAI
import tiktoken
import streamlit as st

# Load API key from Streamlit secrets
op_key = st.secrets["api_keys"]["openai_key"]
client = OpenAI(api_key=op_key)

DATA_DIR = os.getcwd()

# Load .txt documents
def load_documents(folder_path: str):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
                filenames.append(filename)
    return documents, filenames

# Split documents into chunks
def split_text(text, max_tokens=200):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

# Generate OpenAI embeddings
def get_embedding(text: str) -> List[float]:
    result = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return result.data[0].embedding

# Build FAISS index
def build_faiss_index(docs: List[str]):
    index = faiss.IndexFlatL2(1536)
    embeddings = [get_embedding(doc) for doc in docs]
    index.add(np.array(embeddings).astype("float32"))
    np.save("embeddings.npy", embeddings)
    return index, embeddings

# Search top-k similar documents
def search_similar_documents(query, docs, embeddings, index, k=3):
    query_emb = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_emb, k)
    return [docs[i] for i in I[0]]

# Generate answer from GPT-4
def generate_answer(context_docs, user_query):
    context = "\n\n---\n\n".join(context_docs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers based on the provided context. Answer the question directly without mentioning the data source."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        temperature=0.6,
        max_tokens=200
    )
    return response.choices[0].message.content

# Streamlit App
st.set_page_config(page_title="Kalorist Q&A", layout="centered")
st.title("Kalorist Question Answering")

# Load and index documents only once
@st.cache_resource(show_spinner="Indexing documents...")
def initialize_index():
    raw_docs, filenames = load_documents(DATA_DIR)
    docs = []
    for doc in raw_docs:
        docs.extend(split_text(doc))

    if os.path.exists("embeddings.npy"):
        embeddings = np.load("embeddings.npy", allow_pickle=True)
        index = faiss.IndexFlatL2(1536)
        index.add(np.array(embeddings).astype("float32"))
    else:
        index, embeddings = build_faiss_index(docs)

    return docs, embeddings, index

docs, embeddings, index = initialize_index()

# Input field
user_query = st.text_input("Ask a question based on the documents:")

if user_query:
    top_docs = search_similar_documents(user_query, docs, embeddings, index)
    answer = generate_answer(top_docs, user_query)

    st.subheader("Answer")
    st.write(answer)

#    with st.expander("Context Used"):
#        for i, doc in enumerate(top_docs):
#            st.markdown(f"**Document {i+1}:**")
#            st.text(doc)
