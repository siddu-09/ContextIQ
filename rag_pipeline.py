from src.document_processor import process_pdf
from src.embeddings import get_embeddings_model, build_vector_store
from src.retriever import HybridRetriever
from src.llm import generate_answer


class RAGPipeline:

    def __init__(self):
        self.docs             = None
        self.chunks           = None
        self.vector_store     = None
        self.hybrid_retriever = None

    def load(self, pdf_path: str, source_name: str, chunk_size: int = 300, chunk_overlap: int = 50):
        new_docs, new_chunks = process_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk in new_chunks:
            chunk.metadata["source"] = source_name

        if self.docs is None:
            self.docs   = new_docs
            self.chunks = new_chunks
        else:
            self.docs   = self.docs + new_docs
            self.chunks = self.chunks + new_chunks

    def build_index(self):
        embeddings            = get_embeddings_model()
        self.vector_store     = build_vector_store(self.chunks, embeddings)
        self.hybrid_retriever = HybridRetriever(self.chunks, self.vector_store)

    def retrieve(self, query: str, k: int = 5, source_filter: str = None):
        if source_filter:
            filtered_chunks = [c for c in self.chunks if c.metadata.get("source") == source_filter]
            temp_retriever  = HybridRetriever(filtered_chunks, self.vector_store)
            return temp_retriever.retrieve(query, k=k)
        return self.hybrid_retriever.retrieve(query, k=k)

    def answer(self, client, query: str, k: int = 5, source_filter: str = None, context_token_limit: int = 1500):
        results, scores, debug_info = self.retrieve(query, k=k, source_filter=source_filter)

        context_chunks = []
        total_chars    = 0
        limit_chars    = context_token_limit * 4

        for doc in results:
            if total_chars + len(doc.page_content) > limit_chars:
                break
            source = doc.metadata.get("source", "unknown")
            page   = doc.metadata.get("page", "?")
            context_chunks.append((doc.page_content, source, page))
            total_chars += len(doc.page_content)

        return generate_answer(client, context_chunks, query), results, scores, debug_info

    def get_sources(self) -> list:
        if not self.chunks:
            return []
        return list({c.metadata.get("source") for c in self.chunks})