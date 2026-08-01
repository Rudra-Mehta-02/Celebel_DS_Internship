"""
End-to-end RAG pipeline — retrieval + generation.

Combines hybrid retrieval with LLM-based answer generation,
source tracking, and latency measurement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.llm.engine import get_llm
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from the RAG pipeline."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    context_text: str = ""
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    provider: str = ""


RAG_SYSTEM_PROMPT = """You are a knowledgeable AI assistant. Answer the user's question 
based on the provided context. Follow these rules:

1. Use ONLY the provided context to answer. If the context doesn't contain 
   the answer, say "I don't have enough information to answer this."
2. Be concise but thorough.
3. Cite your sources by mentioning where the information comes from.
4. Do not make up information that isn't in the context.
"""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer based on the context above:"""


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant chunks using hybrid search."""
    store = get_vector_store()
    return store.hybrid_search(query, top_k=top_k)


def generate_answer(
    query: str,
    context_chunks: list[dict],
    system_prompt: str = RAG_SYSTEM_PROMPT,
) -> RAGResult:
    """
    Generate an answer from retrieved context chunks.

    Args:
        query: User's question.
        context_chunks: Retrieved chunks from hybrid search.
        system_prompt: System prompt for the LLM.

    Returns:
        RAGResult with answer, sources, and timing.
    """
    llm = get_llm()

    # Assemble context
    context_parts = []
    sources = []
    for i, chunk in enumerate(context_chunks):
        source = chunk.get("metadata", {}).get("source", "unknown")
        title = chunk.get("metadata", {}).get("title", "")
        context_parts.append(f"[Source {i+1}: {title or source}]\n{chunk['text']}")
        sources.append({
            "source": source,
            "title": title,
            "chunk_id": chunk.get("id", ""),
            "score": chunk.get("rrf_score", chunk.get("score", 0)),
        })

    context_text = "\n\n---\n\n".join(context_parts)

    # Generate answer
    prompt = RAG_USER_TEMPLATE.format(context=context_text, question=query)

    t0 = time.time()
    response = llm.generate(prompt, system=system_prompt, temperature=0.3)
    gen_latency = (time.time() - t0) * 1000

    return RAGResult(
        answer=response.text,
        sources=sources,
        context_text=context_text,
        generation_latency_ms=gen_latency,
        provider=response.provider,
    )


def ask(query: str, top_k: int = 5) -> RAGResult:
    """
    Full RAG pipeline: retrieve + generate.

    Args:
        query: User's question.
        top_k: Number of chunks to retrieve.

    Returns:
        RAGResult with answer, sources, and timing.
    """
    # Retrieve
    t0 = time.time()
    chunks = retrieve(query, top_k=top_k)
    retrieval_latency = (time.time() - t0) * 1000

    if not chunks:
        return RAGResult(
            answer="I couldn't find any relevant information in the knowledge base.",
            retrieval_latency_ms=retrieval_latency,
        )

    # Generate
    result = generate_answer(query, chunks)
    result.retrieval_latency_ms = retrieval_latency

    logger.info(
        "RAG: retrieved %d chunks in %.0fms, generated in %.0fms via %s",
        len(chunks), retrieval_latency, result.generation_latency_ms, result.provider,
    )

    return result
