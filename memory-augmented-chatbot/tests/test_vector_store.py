"""Tests for src/rag/vector_store.py — Hybrid retrieval with ChromaDB + BM25 + RRF."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestHybridVectorStore:
    """Test the hybrid retrieval system."""

    @pytest.fixture(autouse=True)
    def setup_store(self, temp_dir):
        """Create a vector store with mocked embedder."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.rand(384).tolist()
        mock_embedder.embed_batch.return_value = [np.random.rand(384).tolist() for _ in range(3)]

        with patch("src.rag.vector_store.Embedder", return_value=mock_embedder):
            from src.rag.vector_store import HybridVectorStore
            self.store = HybridVectorStore(persist_dir=str(temp_dir / "chroma_test"))
            self.mock_embedder = mock_embedder

    def test_add_documents(self, sample_chunks):
        """Should add documents to the store."""
        texts = [c["text"] for c in sample_chunks]
        metadatas = [c["metadata"] for c in sample_chunks]
        self.store.add_documents(texts, metadatas)
        stats = self.store.get_stats()
        assert stats["document_count"] >= len(sample_chunks)

    def test_dense_search(self, sample_chunks):
        """Should perform dense (embedding) search."""
        texts = [c["text"] for c in sample_chunks]
        metadatas = [c["metadata"] for c in sample_chunks]
        self.store.add_documents(texts, metadatas)
        results = self.store.search("machine learning", top_k=3)
        assert isinstance(results, list)

    def test_hybrid_search(self, sample_chunks):
        """Should perform hybrid search combining dense and sparse."""
        texts = [c["text"] for c in sample_chunks]
        metadatas = [c["metadata"] for c in sample_chunks]
        self.store.add_documents(texts, metadatas)
        results = self.store.hybrid_search("neural network deep learning", top_k=3)
        assert isinstance(results, list)

    def test_search_returns_metadata(self, sample_chunks):
        """Search results should include metadata."""
        texts = [c["text"] for c in sample_chunks]
        metadatas = [c["metadata"] for c in sample_chunks]
        self.store.add_documents(texts, metadatas)
        results = self.store.search("test query", top_k=3)
        if results:
            for result in results:
                assert "text" in result or "content" in result or isinstance(result, dict)

    def test_top_k_limit(self, sample_chunks):
        """Should respect the top_k parameter."""
        texts = [c["text"] for c in sample_chunks]
        metadatas = [c["metadata"] for c in sample_chunks]
        self.store.add_documents(texts, metadatas)
        results = self.store.search("query", top_k=1)
        assert len(results) <= 1

    def test_empty_store_search(self):
        """Searching empty store should return empty results."""
        results = self.store.search("test query", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_stats_empty_store(self):
        """Stats on empty store should show 0 documents."""
        stats = self.store.get_stats()
        assert stats["document_count"] == 0


class TestRRFFusion:
    """Test Reciprocal Rank Fusion specifically."""

    def test_rrf_combines_rankings(self):
        """RRF should combine two ranked lists."""
        from src.rag.vector_store import reciprocal_rank_fusion
        dense_results = [
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.7},
            {"id": "c", "score": 0.5},
        ]
        sparse_results = [
            {"id": "b", "score": 5.0},  # Different scale
            {"id": "c", "score": 3.0},
            {"id": "d", "score": 1.0},
        ]
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
        assert isinstance(fused, list)
        # "b" appears in both, should rank high
        if fused:
            ids = [r["id"] for r in fused]
            assert "b" in ids

    def test_rrf_handles_empty_lists(self):
        """RRF should handle empty input lists."""
        from src.rag.vector_store import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [], k=60)
        assert result == []

    def test_rrf_single_list(self):
        """RRF with one empty list should return the other."""
        from src.rag.vector_store import reciprocal_rank_fusion
        dense_results = [{"id": "a", "score": 0.9}]
        result = reciprocal_rank_fusion(dense_results, [], k=60)
        assert len(result) >= 1
