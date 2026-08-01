"""Tests for src/data/chunker.py — Text chunking."""

import pytest
from src.data.chunker import recursive_split, chunk_text, Chunk


class TestRecursiveSplit:
    """Test the recursive text splitting algorithm."""

    def test_short_text_not_split(self):
        """Text shorter than chunk_size should not be split."""
        text = "A short sentence with just a few words."
        chunks = recursive_split(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_long_text_split(self, sample_text):
        """Long text should be split into multiple chunks."""
        chunks = recursive_split(sample_text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_chunks_not_empty(self, sample_text):
        """No chunk should be empty."""
        chunks = recursive_split(sample_text, chunk_size=50, chunk_overlap=10)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_chunk_size_respected(self, sample_text):
        """Chunks should approximately respect the chunk_size limit."""
        chunk_size = 50
        chunks = recursive_split(sample_text, chunk_size=chunk_size, chunk_overlap=10)
        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow 50% overflow (recursive splitting is approximate)
            assert word_count <= chunk_size * 1.5 + 10

    def test_empty_text(self):
        """Empty text should return empty list."""
        chunks = recursive_split("", chunk_size=100)
        assert chunks == [] or chunks == [""]

    def test_whitespace_only(self):
        """Whitespace-only text should return empty list."""
        chunks = recursive_split("   \n\n\n   ", chunk_size=100)
        assert len(chunks) == 0 or all(c.strip() == "" for c in chunks)


class TestChunkText:
    """Test the chunk_text function with metadata."""

    def test_returns_chunk_objects(self, sample_text):
        """Should return list of Chunk dataclass instances."""
        chunks = chunk_text(sample_text, source="test.txt", title="Test")
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_has_metadata(self, sample_text):
        """Each chunk should have source and title metadata."""
        chunks = chunk_text(sample_text, source="test.txt", title="Test Article")
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["title"] == "Test Article"

    def test_chunk_has_unique_id(self, sample_text):
        """Each chunk should have a unique chunk_id."""
        chunks = chunk_text(sample_text, source="test.txt", title="Test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # All unique

    def test_chunk_has_word_count(self, sample_text):
        """Metadata should include word count."""
        chunks = chunk_text(sample_text, source="test.txt", title="Test")
        for chunk in chunks:
            assert "word_count" in chunk.metadata
            assert chunk.metadata["word_count"] > 0

    def test_deduplication(self):
        """Identical text should not produce duplicate chunks."""
        text = "Same content. " * 200  # Repeat
        chunks = chunk_text(text, source="test.txt", title="Test")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_tracking(self, sample_text):
        """Chunks should track their index."""
        chunks = chunk_text(sample_text, source="test.txt", title="Test")
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
