"""
Data ingestion script — run the full pipeline from the command line.

Usage:
    python scripts/ingest.py --file urls.txt
    python scripts/ingest.py https://en.wikipedia.org/wiki/Deep_learning
    python scripts/ingest.py --pdf document.pdf
    python scripts/ingest.py --file urls.txt --skip-kg
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BASE_DIR
from src.data.scraper import scrape_urls, load_urls_from_file
from src.data.cleaner import clean_all
from src.data.chunker import chunk_directory, chunk_text
from src.data.loader import load_pdf, load_url
from src.rag.vector_store import get_vector_store
from src.graph.extractor import extract_from_chunks
from src.graph.store import get_graph_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest_urls(urls: list[str], skip_kg: bool = False) -> dict:
    """Full URL ingestion pipeline."""
    t0 = time.time()

    # Scrape
    logger.info("🌐 Scraping %d URLs...", len(urls))
    scraped = scrape_urls(urls)

    # Clean
    logger.info("🧹 Cleaning HTML...")
    cleaned = clean_all()

    # Chunk
    logger.info("✂️ Chunking text...")
    chunks = chunk_directory()

    # Index
    logger.info("📦 Adding to vector store...")
    store = get_vector_store()
    store.add_chunks(chunks)

    # KG extraction
    entities_count = 0
    if not skip_kg:
        logger.info("🕸️ Extracting knowledge graph...")
        chunk_dicts = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        extractions = extract_from_chunks(chunk_dicts, sample_rate=0.3)
        graph = get_graph_store()
        for ext in extractions:
            graph.add_extraction(ext)
            entities_count += len(ext.get("entities", []))

    elapsed = time.time() - t0

    result = {
        "pages_scraped": len(scraped),
        "pages_cleaned": len(cleaned),
        "chunks_created": len(chunks),
        "entities_extracted": entities_count,
        "elapsed_seconds": round(elapsed, 1),
    }

    logger.info("=" * 50)
    logger.info("✅ Ingestion Complete!")
    for k, v in result.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 50)

    return result


def ingest_pdf(filepath: str, skip_kg: bool = False) -> dict:
    """Ingest a PDF document."""
    doc = load_pdf(filepath)
    if not doc:
        logger.error("Failed to load PDF: %s", filepath)
        return {}

    chunks = chunk_text(doc["text"], source=filepath, title=doc.get("title", ""))
    store = get_vector_store()
    store.add_chunks(chunks)

    logger.info("✅ Ingested PDF: %d chunks from %s", len(chunks), filepath)
    return {"chunks_created": len(chunks), "source": filepath}


def main():
    parser = argparse.ArgumentParser(description="Data ingestion pipeline")
    parser.add_argument("urls", nargs="*", help="URLs to scrape")
    parser.add_argument("--file", "-f", help="File containing URLs (one per line)")
    parser.add_argument("--pdf", help="PDF file to ingest")
    parser.add_argument("--skip-kg", action="store_true", help="Skip knowledge graph extraction")

    args = parser.parse_args()

    if args.pdf:
        ingest_pdf(args.pdf, skip_kg=args.skip_kg)
    elif args.file:
        urls = load_urls_from_file(args.file)
        ingest_urls(urls, skip_kg=args.skip_kg)
    elif args.urls:
        ingest_urls(args.urls, skip_kg=args.skip_kg)
    else:
        # Default: use urls.txt
        url_file = BASE_DIR / "urls.txt"
        if url_file.exists():
            urls = load_urls_from_file(url_file)
            ingest_urls(urls, skip_kg=args.skip_kg)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
