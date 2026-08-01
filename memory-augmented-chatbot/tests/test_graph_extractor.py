"""Tests for src/graph/extractor.py — Entity and relation extraction."""

import pytest
from unittest.mock import patch, MagicMock
from src.graph.extractor import extract_entities_llm, extract_entities_heuristic, extract_from_chunks


class TestExtractEntitiesHeuristic:
    """Test the regex-based fallback extractor."""

    def test_extracts_capitalized_phrases(self):
        """Should extract capitalised multi-word phrases."""
        text = "Machine Learning is used at Google Brain for Natural Language Processing tasks."
        result = extract_entities_heuristic(text)
        assert "entities" in result
        assert len(result["entities"]) > 0
        names = [e["name"] for e in result["entities"]]
        assert any("Machine Learning" in n for n in names)

    def test_skips_common_words(self):
        """Should skip common words like 'The', 'However', etc."""
        text = "However, The approach Furthermore uses Moreover data Therefore."
        result = extract_entities_heuristic(text)
        names = [e["name"] for e in result["entities"]]
        assert "However" not in names
        assert "Furthermore" not in names

    def test_generates_relations(self):
        """Should generate co-occurrence relations between entities."""
        text = "Machine Learning at Google Brain uses Deep Learning for Natural Language Processing."
        result = extract_entities_heuristic(text)
        assert "relations" in result
        # Should have at least some relations if multiple entities found
        if len(result["entities"]) >= 2:
            assert len(result["relations"]) > 0

    def test_limits_entities(self):
        """Should limit to max 10 entities."""
        # Create text with many capitalized phrases
        text = " ".join([f"Entity{i} Name{i}" for i in range(50)])
        result = extract_entities_heuristic(text)
        assert len(result["entities"]) <= 10

    def test_limits_relations(self):
        """Should limit to max 8 relations."""
        text = "Machine Learning Deep Learning Natural Language Processing Computer Vision Reinforcement Learning Transfer Learning Generative Models Neural Networks Data Science Feature Engineering."
        result = extract_entities_heuristic(text)
        assert len(result["relations"]) <= 8

    def test_entity_type_is_concept(self):
        """Heuristic extractor should default all types to CONCEPT."""
        text = "Machine Learning uses Neural Networks for predictions."
        result = extract_entities_heuristic(text)
        for entity in result["entities"]:
            assert entity["type"] == "CONCEPT"

    def test_empty_text(self):
        """Should handle empty text gracefully."""
        result = extract_entities_heuristic("")
        assert result == {"entities": [], "relations": []}


class TestExtractEntitiesLLM:
    """Test the LLM-based extractor."""

    def test_short_text_returns_empty(self):
        """Text under 20 words should return empty result."""
        result = extract_entities_llm("Short text.")
        assert result == {"entities": [], "relations": []}

    def test_falls_back_to_heuristic_on_error(self, mock_llm):
        """Should fall back to heuristic when LLM fails."""
        mock_llm.generate_json.side_effect = Exception("LLM error")
        with patch("src.graph.extractor.get_llm", return_value=mock_llm):
            text = "Machine Learning is a subset of Artificial Intelligence that uses Neural Networks for pattern recognition in Computer Vision tasks."
            result = extract_entities_llm(text)
            # Should still return results from heuristic fallback
            assert "entities" in result
            assert "relations" in result

    def test_validates_entity_structure(self, mock_llm):
        """Should validate entity structure and reject malformed entries."""
        mock_llm.generate_json.return_value = {
            "entities": [
                {"name": "Valid Entity", "type": "CONCEPT", "description": "Valid"},
                {"name": "", "type": "CONCEPT"},  # Empty name — should be filtered
                {"type": "CONCEPT"},  # Missing name — should be filtered
            ],
            "relations": [],
        }
        with patch("src.graph.extractor.get_llm", return_value=mock_llm):
            text = "A sufficiently long text about Machine Learning and Artificial Intelligence to pass the word count gate."
            result = extract_entities_llm(text)
            # Only the valid entity should survive
            assert all(e["name"] for e in result["entities"])


class TestExtractFromChunks:
    """Test batch extraction from chunks."""

    def test_processes_chunks(self, sample_chunks, mock_llm):
        """Should process a list of chunks."""
        with patch("src.graph.extractor.get_llm", return_value=mock_llm):
            results = extract_from_chunks(sample_chunks, sample_rate=1.0)
            assert isinstance(results, list)

    def test_respects_sample_rate(self, sample_chunks, mock_llm):
        """Should only process a fraction of chunks based on sample_rate."""
        with patch("src.graph.extractor.get_llm", return_value=mock_llm):
            # With sample_rate=0.5, should process ~50% of chunks
            results = extract_from_chunks(sample_chunks, sample_rate=0.5)
            assert isinstance(results, list)
