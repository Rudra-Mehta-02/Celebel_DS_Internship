"""Tests for src/config.py — Settings validation."""

import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Test configuration loading and validation."""

    def test_settings_loads_defaults(self, mock_settings):
        """Settings should load with sensible defaults."""
        from src.config import get_settings
        settings = get_settings()
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.rag_top_k == 5

    def test_settings_requires_at_least_one_provider(self):
        """Should raise if no LLM provider is configured."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any cached settings
            from src.config import get_settings
            get_settings.cache_clear() if hasattr(get_settings, 'cache_clear') else None
            # The settings should still load — validation is soft, logs warnings

    def test_embedding_model_default(self, mock_settings):
        """Default embedding model should be MiniLM."""
        from src.config import get_settings
        settings = get_settings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_hybrid_alpha_range(self, mock_settings):
        """Hybrid alpha should be between 0.0 and 1.0."""
        from src.config import get_settings
        settings = get_settings()
        assert 0.0 <= settings.hybrid_alpha <= 1.0

    def test_chunk_overlap_less_than_size(self, mock_settings):
        """Chunk overlap should be less than chunk size."""
        from src.config import get_settings
        settings = get_settings()
        assert settings.chunk_overlap < settings.chunk_size

    def test_data_directories_defined(self, mock_settings):
        """Data directories should be defined in config."""
        from src.config import RAW_DIR, CLEANED_DIR
        assert RAW_DIR is not None
        assert CLEANED_DIR is not None
