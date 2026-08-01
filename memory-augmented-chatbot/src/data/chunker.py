"""
Text chunker — splits cleaned text into retrieval-friendly chunks.

Features:
  - Recursive character splitting (paragraph → sentence → word)
  - Configurable chunk size and overlap
  - Metadata preservation per chunk
  - Near-duplicate detection via simple hashing
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import get_settings, CLEANED_DIR

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:12]


def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[list[str]] = None,
) -> list[str]:
    """
    Recursively split text, trying larger separators first.

    Tries to split by: double-newline → single-newline → period → space → character.
    This keeps paragraphs and sentences intact where possible.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    final_chunks: list[str] = []

    # Count words
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Find the first separator that actually splits the text
    separator = separators[0]
    remaining_separators = separators[1:]

    parts = text.split(separator) if separator else list(text)

    current_chunk_words: list[str] = []
    current_word_count = 0

    for part in parts:
        part_words = part.split()
        part_word_count = len(part_words)

        if current_word_count + part_word_count > chunk_size and current_chunk_words:
            # Emit current chunk
            chunk_text = separator.join(current_chunk_words) if separator else "".join(current_chunk_words)
            chunk_text = chunk_text.strip()
            if chunk_text:
                final_chunks.append(chunk_text)

            # Keep overlap
            overlap_text = separator.join(current_chunk_words) if separator else "".join(current_chunk_words)
            overlap_words = overlap_text.split()
            if len(overlap_words) > chunk_overlap:
                overlap_text = " ".join(overlap_words[-chunk_overlap:])
            current_chunk_words = [overlap_text, part] if overlap_text else [part]
            current_word_count = len(overlap_text.split()) + part_word_count
        else:
            current_chunk_words.append(part)
            current_word_count += part_word_count

    # Emit the last chunk
    if current_chunk_words:
        chunk_text = separator.join(current_chunk_words) if separator else "".join(current_chunk_words)
        chunk_text = chunk_text.strip()
        if chunk_text:
            final_chunks.append(chunk_text)

    # If any chunk is still too large, split with next separator
    if remaining_separators:
        refined: list[str] = []
        for chunk in final_chunks:
            if len(chunk.split()) > chunk_size * 1.5:
                refined.extend(
                    recursive_split(chunk, chunk_size, chunk_overlap, remaining_separators)
                )
            else:
                refined.append(chunk)
        return refined

    return final_chunks


def chunk_text(
    text: str,
    source: str = "",
    title: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Chunk]:
    """
    Split text into chunks with metadata.

    Args:
        text: The text to chunk.
        source: Source URL or file path.
        title: Document title.
        chunk_size: Words per chunk (default from config).
        chunk_overlap: Overlap words (default from config).

    Returns:
        List of Chunk objects with metadata.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    raw_chunks = recursive_split(text, chunk_size, chunk_overlap)

    # Deduplicate by hash
    seen_hashes: set[str] = set()
    chunks: list[Chunk] = []

    for i, chunk_text_str in enumerate(raw_chunks):
        h = hashlib.md5(chunk_text_str.encode()).hexdigest()[:12]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        chunks.append(Chunk(
            text=chunk_text_str,
            metadata={
                "source": source,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(raw_chunks),
                "word_count": len(chunk_text_str.split()),
            },
            chunk_id=h,
        ))

    logger.info(
        "Chunked '%s' → %d chunks (avg %d words)",
        title[:40] or source[:40],
        len(chunks),
        sum(len(c.text.split()) for c in chunks) // max(len(chunks), 1),
    )
    return chunks


def chunk_directory(input_dir: Path = CLEANED_DIR) -> list[Chunk]:
    """Chunk all cleaned text files in a directory."""
    all_chunks: list[Chunk] = []
    txt_files = list(input_dir.glob("*.txt"))

    for filepath in txt_files:
        text = filepath.read_text(encoding="utf-8")
        # Extract title from first line
        title = text.split("\n")[0].strip()[:120] if text else filepath.stem
        chunks = chunk_text(text, source=filepath.name, title=title)
        all_chunks.extend(chunks)

    logger.info("Total: %d chunks from %d files", len(all_chunks), len(txt_files))
    return all_chunks
