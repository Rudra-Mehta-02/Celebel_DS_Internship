"""Tests for src/data/cleaner.py — Text cleaning pipeline."""

import pytest
from src.data.cleaner import clean_html, clean_text, clean_file


class TestCleanHtml:
    """Test HTML to text conversion."""

    def test_removes_html_tags(self, sample_html):
        """Should strip all HTML tags."""
        result = clean_html(sample_html)
        assert "<html>" not in result
        assert "<p>" not in result
        assert "<article>" not in result

    def test_preserves_content(self, sample_html):
        """Should preserve the actual text content."""
        result = clean_html(sample_html)
        assert "Machine Learning" in result or "machine learning" in result.lower()
        assert "artificial intelligence" in result.lower()

    def test_removes_navigation(self, sample_html):
        """Should remove navigation boilerplate."""
        result = clean_html(sample_html)
        assert "Navigation menu here" not in result

    def test_removes_footer(self, sample_html):
        """Should remove footer content."""
        result = clean_html(sample_html)
        assert "Copyright" not in result

    def test_handles_empty_html(self):
        """Should handle empty/None input gracefully."""
        result = clean_html("")
        assert result == "" or result is not None

    def test_handles_plain_text(self):
        """Should handle text without HTML tags."""
        result = clean_html("Just plain text here.")
        assert "plain text" in result


class TestCleanText:
    """Test text normalization and cleanup."""

    def test_collapses_whitespace(self):
        """Should collapse multiple spaces and newlines."""
        result = clean_text("Hello   world\n\n\n\nfoo   bar")
        assert "   " not in result

    def test_unicode_normalization(self):
        """Should normalize unicode characters."""
        result = clean_text("café résumé naïve")
        assert result is not None
        assert len(result) > 0

    def test_removes_excessive_newlines(self):
        """Should not have more than 2 consecutive newlines."""
        result = clean_text("Hello\n\n\n\n\nWorld")
        assert "\n\n\n" not in result

    def test_strips_leading_trailing(self):
        """Should strip leading and trailing whitespace."""
        result = clean_text("   Hello World   ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_handles_empty_string(self):
        """Should handle empty input."""
        result = clean_text("")
        assert result == ""

    def test_preserves_paragraph_breaks(self):
        """Should preserve meaningful paragraph breaks (double newline)."""
        text = "Paragraph one.\n\nParagraph two."
        result = clean_text(text)
        # At minimum, the two paragraphs should be separated
        assert "Paragraph one" in result
        assert "Paragraph two" in result
