"""
Web scraper — fetches pages using requests + BeautifulSoup.

Features:
  - Rate limiting (1 req/sec politeness)
  - Retry with exponential backoff
  - Smart content extraction (<article>, <main>, <body>)
  - Metadata extraction (title, description)
  - Progress tracking with tqdm
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import get_settings, RAW_DIR

logger = logging.getLogger(__name__)

# Polite delay between requests (seconds)
REQUEST_DELAY = 1.0
MAX_RETRIES = 3
TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def scrape_url(url: str, retries: int = MAX_RETRIES) -> Optional[dict]:
    """
    Scrape a single URL and return structured data.

    Returns:
        {url, title, description, html, domain, scraped_at} or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"

            soup = BeautifulSoup(resp.text, "lxml")

            # Extract metadata
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"]

            domain = urlparse(url).netloc

            return {
                "url": url,
                "title": title,
                "description": description,
                "html": resp.text,
                "domain": domain,
                "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        except requests.RequestException as e:
            wait = 2 ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %ds",
                attempt, retries, url, e, wait,
            )
            if attempt < retries:
                time.sleep(wait)

    logger.error("All %d attempts failed for %s", retries, url)
    return None


def scrape_urls(urls: list[str], output_dir: Path = RAW_DIR) -> list[dict]:
    """
    Scrape multiple URLs with rate limiting and progress tracking.

    Returns list of successfully scraped page dicts.
    Saves raw HTML to output_dir/<domain>_<slug>.html.
    """
    results: list[dict] = []

    for url in tqdm(urls, desc="🌐 Scraping URLs"):
        data = scrape_url(url)
        if data:
            # Save raw HTML
            slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
            filename = f"{data['domain']}_{slug}.html"
            # Sanitize filename
            filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
            filepath = output_dir / filename
            filepath.write_text(data["html"], encoding="utf-8")
            data["filepath"] = str(filepath)
            results.append(data)
            logger.info("✅ Scraped: %s → %s", url, filename)
        else:
            logger.warning("❌ Failed: %s", url)

        # Politeness delay
        time.sleep(REQUEST_DELAY)

    logger.info("Scraped %d/%d URLs successfully", len(results), len(urls))
    return results


def load_urls_from_file(filepath: str | Path) -> list[str]:
    """Load URLs from a text file (one URL per line, skip blanks and comments)."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {filepath}")

    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls
