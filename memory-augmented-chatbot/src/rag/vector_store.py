"""
Hybrid vector store — ChromaDB (dense) + BM25 (sparse) + RRF fusion.

This is a key differentiator: neither previous project combines
dense and sparse retrieval with Reciprocal Rank Fusion.

RRF_score(doc) = 1/(k + rank_dense) + 1/(k + rank_sparse)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config import get_settings, CHROMA_DIR
from src.data.chunker import Chunk

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original paper)
RRF_K = 60


class HybridVectorStore:
    """
    ChromaDB for dense retrieval + BM25 for sparse retrieval,
    fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        collection_name: str = "knowledge_base",
        persist_dir: Optional[Path] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir or CHROMA_DIR

        # Initialise ChromaDB
        import chromadb
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index (built lazily)
        self._bm25_corpus: list[dict] = []  # [{id, text, metadata}]
        self._bm25_index = None
        self._bm25_dirty = True

        # Load existing documents for BM25
        self._rebuild_bm25_from_chroma()

        logger.info(
            "✅ HybridVectorStore ready: %d documents in '%s'",
            self._collection.count(), self.collection_name,
        )

    def _rebuild_bm25_from_chroma(self) -> None:
        """Rebuild BM25 index from existing ChromaDB documents."""
        count = self._collection.count()
        if count == 0:
            self._bm25_corpus = []
            self._bm25_index = None
            return

        # Fetch all documents from ChromaDB
        results = self._collection.get(
            include=["documents", "metadatas"],
            limit=count,
        )

        self._bm25_corpus = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                self._bm25_corpus.append({
                    "id": results["ids"][i],
                    "text": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        self._build_bm25()

    def _build_bm25(self) -> None:
        """Build/rebuild the BM25 index from corpus."""
        if not self._bm25_corpus:
            self._bm25_index = None
            return

        from rank_bm25 import BM25Okapi

        tokenised = [doc["text"].lower().split() for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenised)
        self._bm25_dirty = False

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Add chunks to both dense (ChromaDB) and sparse (BM25) indexes.

        Returns number of chunks added.
        """
        if not chunks:
            return 0

        from src.rag.embedder import embed_texts

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Generate embeddings
        embeddings = embed_texts(texts)

        # Upsert into ChromaDB
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        # Update BM25 corpus
        for chunk in chunks:
            self._bm25_corpus.append({
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })
        self._bm25_dirty = True

        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def _dense_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k via ChromaDB cosine similarity."""
        from src.rag.embedder import embed_query

        query_embedding = embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                })
        return hits

    def _sparse_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve top-k via BM25 keyword matching."""
        if self._bm25_dirty:
            self._build_bm25()

        if self._bm25_index is None or not self._bm25_corpus:
            return []

        tokenised_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenised_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        hits = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self._bm25_corpus[idx]
                hits.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(scores[idx]),
                })
        return hits

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: Optional[float] = None,
    ) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        Combines dense (semantic) and sparse (keyword) retrieval.
        alpha controls the weight: 1.0 = pure dense, 0.0 = pure sparse.
        """
        settings = get_settings()
        alpha = alpha if alpha is not None else settings.hybrid_alpha

        # Fetch more candidates than needed for fusion
        fetch_k = top_k * 3

        dense_hits = self._dense_search(query, top_k=fetch_k)
        sparse_hits = self._sparse_search(query, top_k=fetch_k)

        # Build rank maps
        dense_ranks = {h["id"]: rank for rank, h in enumerate(dense_hits)}
        sparse_ranks = {h["id"]: rank for rank, h in enumerate(sparse_hits)}

        # Collect all unique document IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Compute RRF scores
        doc_map: dict[str, dict] = {}
        for h in dense_hits + sparse_hits:
            if h["id"] not in doc_map:
                doc_map[h["id"]] = h

        rrf_scores: list[tuple[str, float]] = []
        for doc_id in all_ids:
            dense_rank = dense_ranks.get(doc_id, fetch_k + 1)
            sparse_rank = sparse_ranks.get(doc_id, fetch_k + 1)

            rrf = (
                alpha * (1.0 / (RRF_K + dense_rank))
                + (1.0 - alpha) * (1.0 / (RRF_K + sparse_rank))
            )
            rrf_scores.append((doc_id, rrf))

        # Sort by RRF score and take top-k
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        top_ids = rrf_scores[:top_k]

        results = []
        for doc_id, score in top_ids:
            doc = doc_map[doc_id]
            doc["rrf_score"] = score
            doc["in_dense"] = doc_id in dense_ranks
            doc["in_sparse"] = doc_id in sparse_ranks
            results.append(doc)

        return results

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Alias for hybrid_search (default retrieval method)."""
        return self.hybrid_search(query, top_k=top_k)

    def count(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the store."""
        self._chroma_client.delete_collection(self.collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_corpus = []
        self._bm25_index = None
        logger.info("Cleared vector store")

    def stats(self) -> dict:
        """Return store statistics."""
        return {
            "collection": self.collection_name,
            "document_count": self.count(),
            "bm25_corpus_size": len(self._bm25_corpus),
            "persist_dir": str(self.persist_dir),
        }


# ── Singleton ────────────────────────────────────────────────
_store: Optional[HybridVectorStore] = None


def get_vector_store() -> HybridVectorStore:
    """Return the cached HybridVectorStore singleton."""
    global _store
    if _store is None:
        _store = HybridVectorStore()
    return _store
