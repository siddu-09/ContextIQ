import warnings
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*torchvision.*")

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_pipeline import RAGPipeline

# -- Env & client --------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -- Page config ---------------------------------------------------------------
st.set_page_config(page_title="ContextIQ", layout="wide")
st.title("ContextIQ")
st.caption("RAG-powered PDF Q&A — Hybrid Search (BM25 + FAISS) via Groq LLM")

# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.header("Retrieval Settings")
    k             = st.slider("Chunks to retrieve (k)", 1, 10, 5)
    chunk_size    = st.slider("Chunk size", 100, 1000, 300, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, step=10)
    context_tokens = st.slider("Context token budget", 500, 4000, 1500, step=100)

# -- File upload ---------------------------------------------------------------
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    cache_key     = (current_names, chunk_size, chunk_overlap)

    if st.session_state.get("cache_key") != cache_key:
        st.session_state.clear()
        st.session_state["cache_key"] = cache_key

        pipeline = RAGPipeline()

        for uploaded_file in uploaded_files:
            path = f"temp_{uploaded_file.name}"
            with open(path, "wb") as f:
                f.write(uploaded_file.read())
            pipeline.load(path, source_name=uploaded_file.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        pipeline.build_index()

        st.session_state["pipeline"] = pipeline
        docs   = pipeline.docs
        chunks = pipeline.chunks
        st.session_state["docs"]   = docs
        st.session_state["chunks"] = chunks

    else:
        pipeline = st.session_state["pipeline"]
        docs     = st.session_state["docs"]
        chunks   = st.session_state["chunks"]

    col1, col2 = st.columns(2)
    col1.metric("Pages loaded", len(docs))
    col2.metric("Chunks created", len(chunks))

    if len(chunks) == 0:
        st.error("No text chunks found. Try another PDF.")
        st.stop()

    with st.expander("Document Preview (first 3 pages)", expanded=False):
        for i, doc in enumerate(docs[:3]):
            st.markdown(f"**Page {i+1}**")
            st.write(doc.page_content)
            st.divider()

    st.success("Both FAISS and BM25 indexes ready — Hybrid search active!")

    sources = pipeline.get_sources()
    source_filter = st.selectbox(
        "Filter by document (optional)",
        options=["All documents"] + sources
    )
    selected_source = None if source_filter == "All documents" else source_filter

    st.divider()
    query = st.text_input("Ask a question about your document", placeholder="e.g. What is the refund policy?")

    if query:
        with st.spinner("Thinking..."):
            answer, results, scores, debug_info = pipeline.answer(
                client,
                query,
                k=k,
                source_filter=selected_source,
                context_token_limit=context_tokens
            )

        st.divider()
        st.subheader("Answer")

        # build tooltip data for each citation number
        tooltips = {}
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "?")
            page_display = page + 1 if isinstance(page, int) else "?"
            snippet = doc.page_content[:200].replace('"', "'").replace("\n", " ")
            tooltips[i] = f"{source} — page {page_display}: {snippet}..."

        import re
        def replace_citation(match):
            num = int(match.group(1))
            tip = tooltips.get(num, "source not found")
            return (
                f'<span style="'
                f'border-bottom: 1px dotted #555;'
                f'cursor: pointer;'
                f'color: #1a73e8;'
                f'font-weight: 500;'
                f'" title="{tip}">[{num}]</span>'
            )

        annotated_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)

        st.markdown(annotated_answer, unsafe_allow_html=True)