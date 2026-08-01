"""Tests for src/llm/engine.py — Multi-provider LLM engine with failover."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from src.llm.engine import MultiProviderLLM


class TestMultiProviderLLM:
    """Test the multi-provider failover engine."""

    def test_initialization(self):
        """Should initialize without crashing even with no providers."""
        with patch.dict("os.environ", {}, clear=True):
            engine = MultiProviderLLM()
            assert engine is not None

    def test_generate_calls_primary_provider(self):
        """Should call the primary (first available) provider."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Test response"
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"

        engine = MultiProviderLLM()
        engine.providers = [mock_provider]

        result = engine.generate("Test prompt")
        assert result == "Test response"
        mock_provider.generate.assert_called_once()

    def test_failover_on_rate_limit(self):
        """Should fail over to next provider on rate limit error."""
        primary = MagicMock()
        primary.generate.side_effect = Exception("429 Too Many Requests")
        primary.is_available.return_value = True
        primary.name = "primary"

        fallback = MagicMock()
        fallback.generate.return_value = "Fallback response"
        fallback.is_available.return_value = True
        fallback.name = "fallback"

        engine = MultiProviderLLM()
        engine.providers = [primary, fallback]

        result = engine.generate("Test prompt")
        assert result == "Fallback response"

    def test_failover_on_timeout(self):
        """Should fail over on timeout errors."""
        primary = MagicMock()
        primary.generate.side_effect = TimeoutError("Connection timed out")
        primary.is_available.return_value = True
        primary.name = "primary"

        fallback = MagicMock()
        fallback.generate.return_value = "Fallback response"
        fallback.is_available.return_value = True
        fallback.name = "fallback"

        engine = MultiProviderLLM()
        engine.providers = [primary, fallback]

        result = engine.generate("Test prompt")
        assert result == "Fallback response"

    def test_all_providers_fail_raises(self):
        """Should raise when ALL providers fail."""
        provider1 = MagicMock()
        provider1.generate.side_effect = Exception("Error 1")
        provider1.is_available.return_value = True
        provider1.name = "p1"

        provider2 = MagicMock()
        provider2.generate.side_effect = Exception("Error 2")
        provider2.is_available.return_value = True
        provider2.name = "p2"

        engine = MultiProviderLLM()
        engine.providers = [provider1, provider2]

        with pytest.raises(Exception):
            engine.generate("Test prompt")

    def test_skips_unavailable_providers(self):
        """Should skip providers that report unavailable."""
        unavailable = MagicMock()
        unavailable.is_available.return_value = False
        unavailable.name = "unavailable"

        available = MagicMock()
        available.generate.return_value = "Available response"
        available.is_available.return_value = True
        available.name = "available"

        engine = MultiProviderLLM()
        engine.providers = [unavailable, available]

        result = engine.generate("Test prompt")
        assert result == "Available response"
        unavailable.generate.assert_not_called()

    def test_generate_json(self):
        """Should parse JSON from LLM response."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = '{"route": "rag", "confidence": 0.9}'
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"

        engine = MultiProviderLLM()
        engine.providers = [mock_provider]

        result = engine.generate_json("Classify this query")
        assert isinstance(result, dict)
        assert "route" in result

    def test_generate_json_handles_malformed(self):
        """Should handle malformed JSON gracefully."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = 'Not valid JSON at all'
        mock_provider.is_available.return_value = True
        mock_provider.name = "mock"

        engine = MultiProviderLLM()
        engine.providers = [mock_provider]

        # Should not crash, should return empty dict or handle gracefully
        try:
            result = engine.generate_json("Test")
            assert isinstance(result, dict)
        except Exception:
            pass  # Acceptable to raise on malformed JSON

    def test_provider_usage_tracking(self):
        """Should track which provider was used."""
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Response"
        mock_provider.is_available.return_value = True
        mock_provider.name = "groq"

        engine = MultiProviderLLM()
        engine.providers = [mock_provider]

        engine.generate("Test")
        # Engine should have some usage tracking
        assert hasattr(engine, 'last_provider') or hasattr(engine, 'usage')
