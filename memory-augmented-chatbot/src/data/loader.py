"""
Document loader — supports PDF, TXT, and URL ingestion.

Provides a unified interface for loading documents from any source
into the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.data.cleaner import extract_text_from_html
from src.data.scraper import scrape_url

logger = logging.getLogger(__name__)


def load_pdf(filepath: str | Path) -> Optional[dict]:
    """
    Extract text from a PDF file.

    Returns:
        {source, title, text, page_count} or None on failure.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("PDF not found: %s", filepath)
        return None

    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(filepath))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            logger.warning("No text extracted from PDF: %s", filepath)
            return None

        return {
            "source": str(filepath),
            "title": filepath.stem,
            "text": full_text,
            "page_count": len(reader.pages),
        }
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", filepath, e)
        return None


def load_txt(filepath: str | Path) -> Optional[dict]:
    """
    Load a plain text file.

    Returns:
        {source, title, text} or None on failure.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("File not found: %s", filepath)
        return None

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return None

        title = text.split("\n")[0].strip()[:120] if text else filepath.stem

        return {
            "source": str(filepath),
            "title": title,
            "text": text,
        }
    except Exception as e:
        logger.error("Failed to read file %s: %s", filepath, e)
        return None


def load_url(url: str) -> Optional[dict]:
    """
    Scrape a URL and extract clean text.

    Returns:
        {source, title, text} or None on failure.
    """
    data = scrape_url(url)
    if not data:
        return None

    text = extract_text_from_html(data["html"])
    if not text.strip():
        logger.warning("No text extracted from URL: %s", url)
        return None

    return {
        "source": url,
        "title": data.get("title", url),
        "text": text,
    }


def load_document(source: str) -> Optional[dict]:
    """
    Auto-detect source type and load accordingly.

    Supports:
      - PDF files (*.pdf)
      - Text files (*.txt)
      - URLs (http:// or https://)
    """
    source_str = str(source).strip()

    if source_str.startswith("http://") or source_str.startswith("https://"):
        return load_url(source_str)
    elif source_str.lower().endswith(".pdf"):
        return load_pdf(source_str)
    elif source_str.lower().endswith(".txt"):
        return load_txt(source_str)
    else:
        # Try as text file
        return load_txt(source_str)
