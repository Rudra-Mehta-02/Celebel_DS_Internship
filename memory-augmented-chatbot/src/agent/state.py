"""
LangGraph agent state definition.
"""

from __future__ import annotations
from typing import TypedDict, Optional


class ChatState(TypedDict, total=False):
    """State that flows through the LangGraph workflow."""

    # ── Input ────────────────────────────────────────────────
    user_id: str
    message: str

    # ── Memory ───────────────────────────────────────────────
    user_facts: list[str]
    chat_history: list[dict]

    # ── Query Processing ─────────────────────────────────────
    rewritten_query: str  # Context-aware rewrite

    # ── Routing ──────────────────────────────────────────────
    route: str  # rag | kg | tool | direct | hybrid
    tool_name: Optional[str]
    tool_args: Optional[str]

    # ── Retrieval ────────────────────────────────────────────
    rag_context: list[dict]  # Chunks with metadata
    kg_context: list[str]  # Graph facts
    tool_result: Optional[str]

    # ── Generation ───────────────────────────────────────────
    answer: str
    sources: list[str]
    confidence: float  # 0.0 to 1.0

    # ── Self-Reflection ──────────────────────────────────────
    reflection_count: int
    needs_retry: bool

    # ── Metadata ─────────────────────────────────────────────
    latency: dict[str, float]  # Per-node timing (ms)
    provider_used: str  # Which LLM provider
    error: Optional[str]
