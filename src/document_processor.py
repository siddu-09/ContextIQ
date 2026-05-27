"""
src/document_processor.py — PDF Loading and Text Chunking for ContextIQ.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(pdf_path: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Loads a PDF document and splits it into text chunks.

    Args:
        pdf_path      : Path to the PDF file
        chunk_size    : Number of characters per text chunk
        chunk_overlap : Overlap between consecutive chunks

    Returns:
        tuple: (docs, chunks)
            - docs: List of loaded Page Documents
            - chunks: List of chunked Document objects
    """
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    
    return docs, chunks
