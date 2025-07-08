from packaging import version
from openai import OpenAI
import os
import faiss
import numpy as np
from typing import List
import tiktoken
import streamlit as st

op_key = st.secrets["api_keys"]["openai_key"]

client = OpenAI(api_key = op_key)

DATA_DIR = os.getcwd()

def load_documents(folder_path: str):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
                filenames.append(filename)
    return documents, filenames

# Split into chunks for embedding (optional: for long files)
def split_text(text, max_tokens=200):
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = enc.encode(text)
    chunks = [enc.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]
    return chunks


# Get OpenAI embeddings
def get_embedding(text: str) -> List[float]:
    result = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return result.data[0].embedding


# Create FAISS index from documents
def build_faiss_index(docs: List[str]):
    index = faiss.IndexFlatL2(1536)
    embeddings = []
    for doc in docs:
        emb = get_embedding(doc)
        print(emb)
        embeddings.append(emb)
    index.add(np.array(embeddings).astype("float32"))

    np.save("index.npy", index)
    
    np.save("embeddings.npy", embeddings)

    return index, embeddings

# Search top-k similar documents
def search_similar_documents(query, docs, embeddings, index, k=3):
    query_emb = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    D, I = index.search(query_emb, k)
    return [docs[i] for i in I[0]]

# Send context + query to GPT-4


def generate_answer(context_docs, user_query):
    context = "\n\n---\n\n".join(context_docs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant who answers based on the provided context. Answer the question directly without mentioning the data source, documents, or context used to generate the answer. Just assume that the context was part of the document that you were being trained on. "},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.6,
        max_tokens=200
    )
    
    #print(response.model)
    
    return response.choices[0].message.content

def main():
    
    print("Loading documents...")
    
    raw_docs, filenames = load_documents(DATA_DIR)

    docs = []
    
    for doc in raw_docs:
        
        docs.extend(split_text(doc))
    
    print("Building vector store...")
    
    if os.path.exists("index.npy"):
        
        if os.path.exists("embeddings.npy"):
        
            print("Index exists, loading it...")
        
            index = np.load("index.npy", allow_pickle=True)
        
            print("Embedding exists, loading it...")
            
            embeddings = np.load("embeddings.npy", allow_pickle=True)
            
            index = faiss.IndexFlatL2(1536)
    
            index.add(np.array(embeddings).astype("float32"))
        
    else:
        
        print("Index and not found, creating a new one...")
        
        index, embeddings = build_faiss_index(docs)

    while True:
        
        query = input("\nEnter your question (or type 'exit'): ")
        
        if query.lower() == "exit":
            
            break

        top_docs = search_similar_documents(query, docs, embeddings, index, k=3)
        
        answer = generate_answer(top_docs, query)
        
        print("\nAnswer:\n", answer)

if __name__ == "__main__":

    main()