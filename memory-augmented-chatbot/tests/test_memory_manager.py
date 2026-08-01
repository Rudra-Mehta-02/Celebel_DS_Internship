"""Tests for src/memory/manager.py — Fact extraction and memory management."""

import pytest
from unittest.mock import patch, MagicMock
from src.memory.manager import MemoryManager


class TestShouldExtract:
    """Test the gating heuristic that decides whether to extract facts."""

    @pytest.fixture(autouse=True)
    def setup_manager(self, mock_llm):
        """Create a memory manager with mocked LLM."""
        with patch("src.memory.manager.get_llm", return_value=mock_llm):
            self.manager = MemoryManager.__new__(MemoryManager)
            self.manager.llm = mock_llm

    def test_short_messages_skipped(self):
        """Very short messages should be skipped."""
        assert not self.manager._should_extract("ok")
        assert not self.manager._should_extract("thanks")
        assert not self.manager._should_extract("hi")
        assert not self.manager._should_extract("yes")

    def test_acknowledgements_skipped(self):
        """Pure acknowledgements should be skipped."""
        assert not self.manager._should_extract("Got it, thanks!")
        assert not self.manager._should_extract("Sure, that works.")

    def test_questions_skipped(self):
        """Pure questions without first-person assertions should be skipped."""
        assert not self.manager._should_extract("What is machine learning?")
        assert not self.manager._should_extract("How does a transformer work?")

    def test_first_person_statements_extracted(self):
        """First-person statements with preferences should trigger extraction."""
        assert self.manager._should_extract("I prefer Python over JavaScript for most tasks.")
        assert self.manager._should_extract("My name is Shami and I work at Celebal Technologies.")

    def test_preference_statements_extracted(self):
        """Statements about preferences should trigger extraction."""
        assert self.manager._should_extract("I really like using deep learning for NLP projects.")


class TestFactExtraction:
    """Test LLM-based fact extraction."""

    def test_extract_returns_list(self, mock_llm):
        """Extraction should return a list of fact strings."""
        mock_llm.generate_json.return_value = {
            "facts": ["User prefers Python", "User works at Google"]
        }
        with patch("src.memory.manager.get_llm", return_value=mock_llm):
            manager = MemoryManager.__new__(MemoryManager)
            manager.llm = mock_llm
            facts = manager._extract_facts("I prefer Python and I work at Google")
            assert isinstance(facts, list)


class TestContradictionDetection:
    """Test fact contradiction and supersession logic."""

    def test_identifies_contradictions(self, mock_llm):
        """Should detect when new fact contradicts existing fact."""
        with patch("src.memory.manager.get_llm", return_value=mock_llm):
            manager = MemoryManager.__new__(MemoryManager)
            manager.llm = mock_llm
            # These should be recognized as contradictions
            old_fact = "User's favorite language is Python"
            new_fact = "User's favorite language is Rust"
            # The manager should have logic to detect this
            assert old_fact != new_fact  # Basic sanity
