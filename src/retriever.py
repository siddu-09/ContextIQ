"""
src/retriever.py — Hybrid Retriever for ContextIQ
Combines BM25 (keyword) + FAISS (semantic) search via Reciprocal Rank Fusion (RRF).
"""

import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """
    Combines BM25 keyword search and FAISS semantic search.
    Results are merged using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, chunks, vector_store, rrf_k: int = 60):
        """
        Args:
            chunks       : List of LangChain Document objects (from text splitter)
            vector_store : FAISS vector store (already built with embeddings)
            rrf_k        : RRF smoothing constant (default 60, standard value)
        """
        self.chunks = chunks
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        # Build BM25 index — tokenize each chunk's text
        tokenized_corpus = [
            chunk.page_content.lower().split() for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _bm25_search(self, query: str, top_n: int) -> list[int]:
        """Returns chunk indices ranked by BM25 score (highest first)."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_n]
        return ranked_indices.tolist()

    def _faiss_search(self, query: str, top_n: int) -> list[int]:
        """Returns chunk indices ranked by FAISS cosine similarity (highest first)."""
        results = self.vector_store.similarity_search(query, k=top_n)
        indices = []
        for result_doc in results:
            for i, chunk in enumerate(self.chunks):
                if chunk.page_content == result_doc.page_content:
                    indices.append(i)
                    break
        return indices

    def _rrf_fusion(self, bm25_ranked: list[int], faiss_ranked: list[int]) -> dict:
        """
        Reciprocal Rank Fusion:
        score(chunk) = 1/(rank_in_bm25 + K) + 1/(rank_in_faiss + K)
        Higher score = more relevant.
        """
        rrf_scores = {}

        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + self.rrf_k)

        for rank, idx in enumerate(faiss_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + self.rrf_k)

        return rrf_scores

    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Main retrieval method. Returns top-k chunks using hybrid search.

        Args:
            query : User's question
            k     : Number of chunks to return

        Returns:
            List of LangChain Document objects, ranked by RRF score
        """
        total_chunks = len(self.chunks)
        search_pool = min(total_chunks, max(k * 4, 20))  # cast wider net before fusion

        bm25_ranked = self._bm25_search(query, top_n=search_pool)
        faiss_ranked = self._faiss_search(query, top_n=search_pool)

        rrf_scores = self._rrf_fusion(bm25_ranked, faiss_ranked)

        # Sort all seen chunks by RRF score descending
        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        top_chunks = [self.chunks[i] for i in sorted_indices[:k]]

        # Also return scores for UI display
        top_scores = [rrf_scores[i] for i in sorted_indices[:k]]

        return top_chunks, top_scores, {
            "bm25_top5": [self.chunks[i].page_content[:80] for i in bm25_ranked[:5]],
            "faiss_top5": [self.chunks[i].page_content[:80] for i in faiss_ranked[:5]],
        }
