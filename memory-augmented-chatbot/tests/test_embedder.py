"""Tests for src/rag/embedder.py — Embedding generation."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


class TestEmbedder:
    """Test the SentenceTransformers embedder wrapper."""

    @pytest.fixture(autouse=True)
    def setup_embedder(self):
        """Create embedder with mocked model to avoid downloading."""
        mock_model = MagicMock()
        # Return fake 384-dim embeddings
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch("src.rag.embedder.SentenceTransformer", return_value=mock_model):
            from src.rag.embedder import Embedder
            self.embedder = Embedder()
            self.mock_model = mock_model

    def test_embed_single_text(self):
        """Should embed a single text string."""
        self.mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        result = self.embedder.embed("Hello world")
        assert result is not None
        assert len(result) == 384

    def test_embed_batch(self):
        """Should embed a batch of texts."""
        texts = ["Hello", "World", "Test"]
        self.mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        results = self.embedder.embed_batch(texts)
        assert len(results) == 3
        assert all(len(r) == 384 for r in results)

    def test_embed_empty_string(self):
        """Should handle empty string gracefully."""
        self.mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        result = self.embedder.embed("")
        assert result is not None

    def test_embeddings_are_floats(self):
        """Embeddings should be float arrays."""
        self.mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        result = self.embedder.embed("Test sentence")
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_dimension_consistency(self):
        """All embeddings should have the same dimension."""
        self.mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
        texts = ["First", "Second", "Third"]
        results = self.embedder.embed_batch(texts)
        dims = [len(r) for r in results]
        assert len(set(dims)) == 1  # All same dimension
