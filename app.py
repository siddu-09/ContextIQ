import warnings
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*torchvision.*")

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from groq import Groq

# ✅ Load env FIRST
load_dotenv()

# ✅ Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("ContextIQ")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # Show content preview
    st.subheader("Document Content Preview")
    for doc in docs[:3]:
        st.write(doc.page_content)

    # 🔥 Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Debug info
    st.write(f"Number of pages loaded: {len(docs)}")

    if len(chunks) > 0:
        st.success(f"Document split into {len(chunks)} chunks")
    else:
        st.warning("No chunks created")

    # Safety check
    if len(chunks) == 0:
        st.error("❌ No text chunks found. Try another PDF.")
        st.stop()

    # 🔥 Embeddings (FREE)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 🔥 Vector DB
    vector_store = FAISS.from_documents(chunks, embeddings)

    st.success("Embeddings created and stored in FAISS")

    # 💬 User question
    query = st.text_input("Ask a question about your document")

    if query:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        results = retriever.get_relevant_documents(query)

        st.subheader("Retrieved Chunks")
        for doc in results[:3]:
            st.write(doc.page_content)

        # Prepare context
        context = "\n\n".join([doc.page_content for doc in results])

        # 🔥 HARD LIMIT
        context = context[:800]

        # 🔥 GROQ LLM (FIXED)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer only using the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
        )

        answer = response.choices[0].message.content

        st.subheader("📌 Answer")
        st.write(answer)