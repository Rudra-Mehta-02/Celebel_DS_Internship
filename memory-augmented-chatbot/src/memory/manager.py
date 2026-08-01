"""
Memory manager — fact extraction, contradiction detection, retrieval.

This is the intelligence layer on top of the raw memory store.
It decides WHAT to remember, detects contradictions, and retrieves
the most relevant facts for a given query.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.llm.engine import get_llm
from src.memory.store import get_memory_store
from src.config import get_settings

logger = logging.getLogger(__name__)

# ── Gating Heuristics ────────────────────────────────────────

# Messages that are too short or trivial to extract facts from
SKIP_PATTERNS = [
    r"^(ok|okay|sure|thanks|thank you|got it|yes|no|yep|nope|hmm|hm|ah|oh)[\.\!\?]?$",
    r"^(hi|hello|hey|howdy|greetings)[\.\!\?]?$",
    r"^(bye|goodbye|see you|later|good night)[\.\!\?]?$",
]

MIN_MESSAGE_LENGTH = 5  # words


def _should_extract(message: str) -> bool:
    """Gate: decide whether a message is worth extracting facts from."""
    message_clean = message.strip().lower()

    # Too short
    if len(message_clean.split()) < MIN_MESSAGE_LENGTH:
        return False

    # Trivial pattern
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, message_clean, re.IGNORECASE):
            return False

    # Pure question (starts with question word, no first-person statements)
    if re.match(r"^(what|where|when|who|how|why|can|could|would|should|is|are|do|does)\b", message_clean):
        # But keep if it also contains first-person info
        if not re.search(r"\b(i am|i'm|my|i have|i've|i like|i prefer|i work|i live|i study)\b", message_clean):
            return False

    return True


# ── Fact Extraction ──────────────────────────────────────────

FACT_EXTRACTION_PROMPT = """Extract durable personal facts from this user message.

Rules:
- Extract ONLY facts about the user (preferences, personal info, interests, goals)
- Each fact should be a self-contained statement starting with "User..."
- Categories: personal_info, preferences, interests, context, goals
- If no personal facts exist, return empty arrays
- Return ONLY valid JSON

Return format:
{{
  "facts": [
    {{"fact": "User's name is Alice", "category": "personal_info", "confidence": 0.95}},
    {{"fact": "User prefers Python", "category": "preferences", "confidence": 0.9}}
  ]
}}

User message: {message}
"""


def extract_facts(message: str) -> list[dict]:
    """
    Extract durable facts from a user message.

    Returns list of {fact, category, confidence} dicts.
    """
    if not _should_extract(message):
        return []

    try:
        llm = get_llm()
        result = llm.generate_json(
            FACT_EXTRACTION_PROMPT.format(message=message),
            temperature=0.0,
            max_tokens=512,
        )

        facts = result.get("facts", [])
        valid = []
        for f in facts:
            if isinstance(f, dict) and "fact" in f:
                fact_text = str(f["fact"]).strip()
                if fact_text and len(fact_text) > 5:
                    valid.append({
                        "fact": fact_text,
                        "category": str(f.get("category", "general")),
                        "confidence": float(f.get("confidence", 0.8)),
                    })
        return valid

    except Exception as e:
        logger.warning("Fact extraction failed: %s", e)
        return []


# ── Contradiction Detection ──────────────────────────────────

CONTRADICTION_CATEGORIES = {
    "personal_info": ["name", "age", "location", "job", "school", "university", "city", "country"],
    "preferences": ["language", "framework", "color", "food", "favorite", "prefer"],
}


def _find_contradictions(user_id: str, new_fact: str) -> list[dict]:
    """
    Find existing facts that might contradict the new fact.

    Simple heuristic: same category + overlapping key words = potential contradiction.
    """
    store = get_memory_store()
    existing = store.get_facts(user_id)

    contradictions = []
    new_lower = new_fact.lower()

    for existing_fact in existing:
        old_lower = existing_fact["fact"].lower()

        # Check for overlapping subject-matter keywords
        for category, keywords in CONTRADICTION_CATEGORIES.items():
            for kw in keywords:
                if kw in new_lower and kw in old_lower:
                    # Same topic mentioned — likely a contradiction or update
                    if new_lower != old_lower:
                        contradictions.append(existing_fact)
                    break

    return contradictions


# ── Public API ───────────────────────────────────────────────

def process_message(user_id: str, role: str, content: str) -> list[dict]:
    """
    Process a message: store in history + extract and persist facts.

    Args:
        user_id: User identifier.
        role: "user" or "assistant".
        content: Message content.

    Returns:
        List of newly stored facts.
    """
    store = get_memory_store()

    # Always store in chat history
    store.add_message(user_id, role, content)

    # Only extract facts from user messages
    if role != "user":
        return []

    if not get_settings().fact_extraction_enabled:
        return []

    # Extract facts
    new_facts = extract_facts(content)
    stored = []

    for f in new_facts:
        # Check for contradictions
        contradictions = _find_contradictions(user_id, f["fact"])

        # Supersede contradicting facts
        for old_fact in contradictions:
            logger.info(
                "Superseding fact '%s' with '%s' for user %s",
                old_fact["fact"], f["fact"], user_id,
            )

        # Store new fact
        fact_id = store.add_fact(
            user_id, f["fact"],
            category=f["category"],
            confidence=f["confidence"],
        )

        if fact_id:
            # Mark old facts as superseded
            for old_fact in contradictions:
                store.deactivate_fact(old_fact["id"], superseded_by=fact_id)

            stored.append(f)
            logger.info("Stored fact for %s: %s", user_id, f["fact"])

    return stored


def get_user_context(user_id: str) -> dict:
    """
    Get the full user context: facts + chat history.

    Returns:
        {facts: [...], history: [...]}
    """
    store = get_memory_store()
    settings = get_settings()

    facts = store.get_facts(user_id, limit=settings.max_memory_facts)
    history = store.get_history(user_id, limit=settings.max_history_turns)

    return {
        "facts": facts,
        "history": history,
    }


def get_user_facts(user_id: str) -> list[dict]:
    """Get all active facts for a user."""
    return get_memory_store().get_facts(user_id)


def clear_user_memory(user_id: str) -> None:
    """Clear all memory for a user."""
    get_memory_store().clear_user(user_id)


def delete_fact(fact_id: str) -> None:
    """Delete a specific fact."""
    get_memory_store().delete_fact(fact_id)
