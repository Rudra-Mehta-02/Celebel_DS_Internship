"""
Embedding generator — wraps SentenceTransformers.

Provides batch embedding with caching and normalisation.
Model auto-downloads on first use (~90 MB for all-MiniLM-L6-v2).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

# Module-level model cache
_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("✅ Embedding model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Embed a list of texts into dense vectors.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        List of embedding vectors (list of floats).
    """
    if not texts:
        return []

    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,  # Normalise for cosine similarity
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = _get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    dot = np.dot(a_np, b_np)
    norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if norm == 0:
        return 0.0
    return float(dot / norm)
