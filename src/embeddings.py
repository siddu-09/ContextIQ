"""
src/embeddings.py — Embedding model and Vector Store utilities for ContextIQ.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings_model(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace embeddings model.

    Args:
        model_name : Name of the HuggingFace model to load

    Returns:
        HuggingFaceEmbeddings: The embedding utility
    """
    return HuggingFaceEmbeddings(model_name=model_name)

def build_vector_store(chunks, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Builds a FAISS vector store from the provided text chunks.

    Args:
        chunks     : List of chunked Document objects
        embeddings : HuggingFaceEmbeddings instance to generate vector embeddings

    Returns:
        FAISS: The initialized FAISS vector store
    """
    return FAISS.from_documents(chunks, embeddings)
