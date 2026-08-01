"""
Centralized configuration using Pydantic BaseSettings.

All settings are loaded from environment variables (or .env file).
Every external service has a local fallback — the system runs
with just ONE free LLM API key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


# ── Project paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
CHROMA_DIR = DATA_DIR / "chroma"
GRAPH_DIR = DATA_DIR / "graph"
MEMORY_DIR = DATA_DIR / "memory"

# Ensure data directories exist
for _d in [RAW_DIR, CLEANED_DIR, CHROMA_DIR, GRAPH_DIR, MEMORY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings — loaded from environment / .env file."""

    # ── LLM Providers ────────────────────────────────────────
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_BASE_URL"
    )

    # ── Model Names ──────────────────────────────────────────
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_fast_model: str = Field(
        default="llama-3.1-8b-instant", alias="GROQ_FAST_MODEL"
    )
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # ── Graph Database ───────────────────────────────────────
    neo4j_uri: Optional[str] = Field(default=None, alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: Optional[str] = Field(default=None, alias="NEO4J_PASSWORD")

    # ── User Database ────────────────────────────────────────
    postgres_dsn: Optional[str] = Field(default=None, alias="POSTGRES_DSN")

    # ── RAG Configuration ────────────────────────────────────
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    hybrid_alpha: float = Field(default=0.7, alias="HYBRID_ALPHA")

    # ── Memory Configuration ─────────────────────────────────
    max_history_turns: int = Field(default=10, alias="MAX_HISTORY_TURNS")
    max_memory_facts: int = Field(default=50, alias="MAX_MEMORY_FACTS")
    fact_extraction_enabled: bool = Field(
        default=True, alias="FACT_EXTRACTION_ENABLED"
    )

    # ── Paths ────────────────────────────────────────────────
    data_dir: Path = DATA_DIR
    chroma_dir: Path = CHROMA_DIR
    graph_dir: Path = GRAPH_DIR
    memory_dir: Path = MEMORY_DIR

    model_config = {
        "env_file": str(BASE_DIR / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    # ── Derived helpers ──────────────────────────────────────

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_gemini(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_neo4j(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)

    @property
    def has_postgres(self) -> bool:
        return bool(self.postgres_dsn)

    @property
    def sqlite_path(self) -> str:
        return str(MEMORY_DIR / "memory.db")

    @property
    def networkx_path(self) -> str:
        return str(GRAPH_DIR / "knowledge_graph.json")

    def validate_llm(self) -> None:
        """Ensure at least one LLM provider is configured."""
        if not (self.has_groq or self.has_gemini):
            raise ValueError(
                "At least one LLM provider must be configured. "
                "Set GROQ_API_KEY or GOOGLE_API_KEY in your .env file.\n"
                "  FREE Groq key: https://console.groq.com/keys\n"
                "  FREE Gemini key: https://aistudio.google.com"
            )


# ── Singleton ────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
