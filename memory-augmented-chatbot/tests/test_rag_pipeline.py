"""Tests for src/rag/pipeline.py — End-to-end RAG pipeline."""

import pytest
from unittest.mock import patch, MagicMock


class TestRAGPipeline:
    """Test the end-to-end RAG pipeline."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Create pipeline with mocked dependencies."""
        self.mock_vector_store = MagicMock()
        self.mock_vector_store.hybrid_search.return_value = [
            {"text": "Machine learning is a subset of AI.", "metadata": {"source": "test.txt"}},
            {"text": "Deep learning uses neural networks.", "metadata": {"source": "test.txt"}},
        ]
        self.mock_vector_store.search.return_value = [
            {"text": "Machine learning is a subset of AI.", "metadata": {"source": "test.txt"}},
        ]

        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = "Machine learning is a subset of AI that enables systems to learn from data."

    def test_retrieve_returns_chunks(self):
        """Retrieval should return relevant chunks."""
        with patch("src.rag.pipeline.get_vector_store", return_value=self.mock_vector_store):
            from src.rag.pipeline import retrieve
            results = retrieve("What is machine learning?")
            assert isinstance(results, list)

    def test_generate_with_context(self):
        """Should generate answer using retrieved context."""
        with patch("src.rag.pipeline.get_vector_store", return_value=self.mock_vector_store), \
             patch("src.rag.pipeline.get_llm", return_value=self.mock_llm):
            from src.rag.pipeline import rag_query
            result = rag_query("What is machine learning?")
            assert isinstance(result, dict)
            assert "answer" in result

    def test_empty_query_handled(self):
        """Empty query should be handled gracefully."""
        with patch("src.rag.pipeline.get_vector_store", return_value=self.mock_vector_store):
            from src.rag.pipeline import retrieve
            results = retrieve("")
            assert isinstance(results, list)

    def test_sources_included(self):
        """Response should include source information."""
        with patch("src.rag.pipeline.get_vector_store", return_value=self.mock_vector_store), \
             patch("src.rag.pipeline.get_llm", return_value=self.mock_llm):
            from src.rag.pipeline import rag_query
            result = rag_query("What is deep learning?")
            assert "sources" in result or "context" in result
