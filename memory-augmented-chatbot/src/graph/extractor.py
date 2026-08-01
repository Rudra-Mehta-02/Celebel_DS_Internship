"""
Entity & relation extractor — LLM-based with heuristic fallback.

Extracts structured {entities, relations} JSON from text chunks
using the LLM, with a regex/NP-based fallback when LLM fails.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.llm.engine import get_llm

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Return a JSON object with this EXACT structure:
{{
  "entities": [
    {{"name": "entity name", "type": "CONCEPT|PERSON|ORG|TECHNOLOGY|ALGORITHM|DATASET|METRIC", "description": "brief description"}}
  ],
  "relations": [
    {{"source": "entity1 name", "target": "entity2 name", "type": "RELATIONSHIP_TYPE", "description": "brief description"}}
  ]
}}

Rules:
- Extract 3-10 entities and 2-8 relationships
- Entity types: CONCEPT, PERSON, ORG, TECHNOLOGY, ALGORITHM, DATASET, METRIC
- Relationship types: USES, PART_OF, DEVELOPED_BY, RELATED_TO, COMPARED_TO, APPLIED_IN, SUBCLASS_OF, TRAINED_ON
- Use the EXACT entity name (properly capitalized) in relationships
- Return ONLY valid JSON, no explanation

Text:
{text}
"""


def extract_entities_llm(text: str) -> dict:
    """
    Extract entities and relations using the LLM.

    Returns:
        {entities: [...], relations: [...]} or empty dict on failure.
    """
    if not text or len(text.split()) < 20:
        return {"entities": [], "relations": []}

    try:
        llm = get_llm()
        prompt = EXTRACTION_PROMPT.format(text=text[:3000])  # Cap text length
        result = llm.generate_json(prompt, temperature=0.0, max_tokens=2048)

        # Validate structure
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Validate entities
        valid_entities = []
        for e in entities:
            if isinstance(e, dict) and "name" in e and "type" in e:
                # Sanitise
                e["name"] = str(e["name"]).strip()
                e["type"] = str(e.get("type", "CONCEPT")).upper()
                e["description"] = str(e.get("description", ""))
                if e["name"]:
                    valid_entities.append(e)

        # Validate relations
        valid_relations = []
        entity_names = {e["name"].lower() for e in valid_entities}
        for r in relations:
            if isinstance(r, dict) and "source" in r and "target" in r and "type" in r:
                r["source"] = str(r["source"]).strip()
                r["target"] = str(r["target"]).strip()
                # Sanitise relationship type (alphanumeric + underscore only)
                r["type"] = re.sub(r"[^A-Z0-9_]", "_", str(r["type"]).upper())
                r["description"] = str(r.get("description", ""))
                if r["source"] and r["target"] and r["type"]:
                    valid_relations.append(r)

        return {"entities": valid_entities, "relations": valid_relations}

    except Exception as e:
        logger.warning("LLM extraction failed: %s — falling back to heuristic", e)
        return extract_entities_heuristic(text)


def extract_entities_heuristic(text: str) -> dict:
    """
    Heuristic fallback — extract entities using regex patterns.

    Looks for capitalised multi-word phrases (likely proper nouns/concepts).
    """
    entities = []
    seen = set()

    # Find capitalised phrases (2-4 words)
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
    matches = re.findall(pattern, text)

    # Common words to skip
    skip_words = {
        "The", "This", "That", "These", "Those", "There", "Here",
        "However", "Although", "Furthermore", "Moreover", "Therefore",
        "In", "On", "At", "By", "For", "With", "From", "About",
    }

    for match in matches:
        normalised = match.strip()
        if (
            normalised.lower() not in seen
            and normalised not in skip_words
            and len(normalised) > 2
            and len(normalised.split()) <= 4
        ):
            seen.add(normalised.lower())
            entities.append({
                "name": normalised,
                "type": "CONCEPT",
                "description": "",
            })

    # Limit to top 10
    entities = entities[:10]

    # Generate simple relations between co-occurring entities
    relations = []
    for i in range(len(entities)):
        for j in range(i + 1, min(i + 3, len(entities))):
            relations.append({
                "source": entities[i]["name"],
                "target": entities[j]["name"],
                "type": "RELATED_TO",
                "description": "co-occurs in text",
            })

    return {"entities": entities, "relations": relations[:8]}


def extract_from_chunks(chunks: list[dict], sample_rate: float = 0.5) -> list[dict]:
    """
    Extract entities and relations from a list of text chunks.

    Args:
        chunks: List of {text, metadata} dicts.
        sample_rate: Fraction of chunks to process (saves API calls).

    Returns:
        List of extraction results with source metadata.
    """
    import random

    results = []
    total = len(chunks)

    # Sample chunks to respect rate limits
    sample_size = max(1, int(total * sample_rate))
    sampled = random.sample(chunks, min(sample_size, total))

    logger.info("Extracting entities from %d/%d chunks", len(sampled), total)

    for i, chunk in enumerate(sampled):
        text = chunk.get("text", "")
        extraction = extract_entities_llm(text)

        if extraction.get("entities"):
            extraction["source"] = chunk.get("metadata", {}).get("source", "")
            results.append(extraction)
            logger.debug(
                "Chunk %d/%d: %d entities, %d relations",
                i + 1, len(sampled),
                len(extraction["entities"]),
                len(extraction["relations"]),
            )

    logger.info(
        "Extraction complete: %d chunks yielded entities",
        len(results),
    )
    return results
