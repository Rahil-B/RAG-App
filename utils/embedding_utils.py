# utils/embedding_utils.py
import streamlit as st
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

# Cache expensive resources using streamlit's caching
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"metric": "cosine"}
    )
    return collection

def add_documents_to_collection(text_chunks, sentence_model):
    collection = get_chroma_collection()
    embeddings = sentence_model.encode(text_chunks).tolist()
    ids = [str(idx) for idx in range(len(text_chunks))]
    collection.add(ids=ids, documents=text_chunks, embeddings=embeddings)

def query_chroma(query, embedding_model, threshold=0.5):
    collection = get_chroma_collection()
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    if not results['documents'][0]:
        return None

    scores = results['distances'][0]  # Lower values mean more similarity.
    valid_contexts = [
        doc for doc, score in zip(results['documents'][0], scores) if score > threshold
    ]
    return valid_contexts if valid_contexts else None
