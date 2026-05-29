---
title: ContextIQ
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# ContextIQ

ContextIQ is a Retrieval-Augmented Generation (RAG) demo built with Streamlit. Upload PDFs and ask natural-language questions about their contents. The app combines a hybrid retrieval strategy (BM25 keyword search + FAISS semantic search) and the Groq LLM to produce answers with inline citations.

Key features

- Upload one or more PDFs and automatically chunk text for retrieval.
- Hybrid retrieval combining BM25 (keyword) and FAISS (semantic) with Reciprocal Rank Fusion (RRF).
- Uses Groq LLM for answer generation with explicit context citation (e.g., [1], [2]).
- Streamlit UI with controls for chunking and retrieval parameters.

Quick start

1. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq API key in a `.env` file at the project root:

   ```text
   GROQ_API_KEY=your_api_key_here
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

Usage

- Open the Streamlit URL shown after launching the app.
- Use the sidebar to tune retrieval and chunking parameters (`k`, chunk size, overlap, token budget).
- Upload PDFs; the app builds BM25 and FAISS indexes and displays pages/chunks processed.
- Ask a question in the text box. The app returns an answer with inline citations; hover citation numbers to view source snippets.

Project layout

- `app.py` — Streamlit frontend and UX logic. See [app.py](app.py).
- `rag_pipeline.py` — Orchestrates ingestion, indexing, retrieval and answer generation. See [rag_pipeline.py](rag_pipeline.py).
- `src/document_processor.py` — PDF loading and chunking using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- `src/embeddings.py` — Embedding model wrapper and FAISS vector store builder.
- `src/retriever.py` — HybridRetriever combines BM25 and FAISS and fuses results with RRF.
- `src/llm.py` — Builds the prompt and calls the Groq client for completion.
- `requirements.txt` — Python dependencies. See [requirements.txt](requirements.txt).

Architecture notes

- Ingestion: PDFs are split into overlapping text chunks (configurable chunk size & overlap).
- Embeddings: Each chunk is embedded using `sentence-transformers` via a LangChain wrapper.
- Vector store: FAISS stores dense vectors for semantic retrieval.
- Retrieval: `HybridRetriever` runs BM25 and FAISS passes and merges results using Reciprocal Rank Fusion for robust ranking.
- Generation: The top context chunks are concatenated, numbered, and provided to the Groq LLM with a system prompt to answer using only the provided context.

Environment & troubleshooting

- Ensure `GROQ_API_KEY` is set in `.env`; the app will stop with an error if missing.
- Installing `faiss-cpu` on macOS can sometimes be tricky; consider using `conda` or follow FAISS docs.
- If you see long load times or high memory usage, reduce `chunk_size` or `k`.

Extending the project

- Swap the embeddings model in `src/embeddings.py` or add other vectorstores.
- Replace the Groq client call in `src/llm.py` with another LLM provider.
- Persist the vector index to disk to avoid rebuilding across runs.

License & contribution
This repository is a demo; contributions and issues are welcome.

Contact
Open an issue if you'd like help deploying or extending the project.
