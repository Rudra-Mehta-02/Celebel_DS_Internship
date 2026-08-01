"""
Text cleaner — converts raw HTML to clean, structured text.

Features:
  - Smart content extraction (article / main content area)
  - Noise removal (nav, footer, sidebar, scripts, styles)
  - Unicode normalisation
  - Whitespace collapse
  - Quality filtering (skip pages with too little text)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Comment

from src.config import RAW_DIR, CLEANED_DIR

logger = logging.getLogger(__name__)

# Tags that are pure noise
NOISE_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "form", "button", "input", "select", "textarea",
    "noscript", "iframe", "svg", "figure", "figcaption",
}

# Minimum text length to keep a page (characters)
MIN_TEXT_LENGTH = 200


def extract_text_from_html(html: str) -> str:
    """
    Extract clean text from HTML, focusing on main content.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # Try to find the main content area
    content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"role": "main"})
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"id": "mw-content-text"})  # Wikipedia
        or soup.find("body")
    )

    if content is None:
        content = soup

    # Extract text, preserving some structure
    text = content.get_text(separator="\n", strip=True)

    # Clean up
    text = _normalise_text(text)

    return text


def _normalise_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, clean up artefacts."""
    # Unicode normalisation
    text = unicodedata.normalize("NFKD", text)

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Collapse multiple blank lines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces to single space (per line)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)

    # Remove citation brackets like [1], [2], [edit]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    return text.strip()


def clean_file(filepath: Path, output_dir: Path = CLEANED_DIR) -> Optional[dict]:
    """
    Clean a single HTML file and save the result.

    Returns:
        {source_file, output_file, title, char_count, word_count} or None if too short.
    """
    html = filepath.read_text(encoding="utf-8", errors="replace")
    text = extract_text_from_html(html)

    if len(text) < MIN_TEXT_LENGTH:
        logger.warning("Skipping %s — too short (%d chars)", filepath.name, len(text))
        return None

    # Extract title from first non-empty line
    title = ""
    for line in text.split("\n"):
        if line.strip():
            title = line.strip()[:120]
            break

    # Save cleaned text
    output_name = filepath.stem + ".txt"
    output_path = output_dir / output_name
    output_path.write_text(text, encoding="utf-8")

    result = {
        "source_file": str(filepath),
        "output_file": str(output_path),
        "title": title,
        "char_count": len(text),
        "word_count": len(text.split()),
    }
    logger.info("✅ Cleaned: %s → %s (%d words)", filepath.name, output_name, result["word_count"])
    return result


def clean_all(input_dir: Path = RAW_DIR, output_dir: Path = CLEANED_DIR) -> list[dict]:
    """Clean all HTML files in input_dir and save to output_dir."""
    results = []
    html_files = list(input_dir.glob("*.html"))

    if not html_files:
        logger.warning("No HTML files found in %s", input_dir)
        return results

    for filepath in html_files:
        result = clean_file(filepath, output_dir)
        if result:
            results.append(result)

    logger.info("Cleaned %d/%d files", len(results), len(html_files))
    return results
