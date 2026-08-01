"""
Evaluation metrics — 4-layer assessment of RAG quality.

Layer 1: Retrieval metrics (automated, free)
Layer 2: Lexical end-to-end metrics (automated, free)
Layer 3: LLM-as-Judge (requires LLM calls)
Layer 4: Memory-specific metrics (unique to us)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Layer 1: Retrieval Metrics ───────────────────────────────

def hit_rate(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Was at least one relevant document retrieved?"""
    if not relevant_ids:
        return 0.0
    return 1.0 if any(r in relevant_ids for r in retrieved_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank — how high was the first relevant document?"""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Precision@k — fraction of retrieved docs that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for d in top_k if d in relevant_ids)
    return relevant_count / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Recall@k — fraction of relevant docs that were retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for d in top_k if d in relevant_ids)
    return relevant_count / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Normalised Discounted Cumulative Gain @ k."""
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # +2 because position is 1-indexed

    # Ideal DCG
    ideal_length = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ── Layer 2: Lexical End-to-End Metrics ──────────────────────

def _tokenise(text: str) -> set[str]:
    """Simple whitespace tokenisation with lowercasing."""
    return set(re.findall(r"\w+", text.lower()))


def groundedness(answer: str, context: str) -> float:
    """% of answer words that appear in the context."""
    answer_words = _tokenise(answer)
    context_words = _tokenise(context)
    if not answer_words:
        return 0.0
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def hallucination_rate(answer: str, context: str) -> float:
    """% of answer words NOT in context (inverse of groundedness)."""
    return 1.0 - groundedness(answer, context)


def context_utilisation(answer: str, context: str) -> float:
    """% of context words used in the answer."""
    answer_words = _tokenise(answer)
    context_words = _tokenise(context)
    if not context_words:
        return 0.0
    overlap = answer_words & context_words
    return len(overlap) / len(context_words)


def answer_relevance(answer: str, question: str) -> float:
    """Word overlap between answer and question."""
    answer_words = _tokenise(answer)
    question_words = _tokenise(question)
    if not question_words:
        return 0.0
    overlap = answer_words & question_words
    return len(overlap) / len(question_words)


def factual_consistency(answer: str, ground_truth: str) -> float:
    """N-gram overlap between answer and ground truth."""
    answer_words = _tokenise(answer)
    truth_words = _tokenise(ground_truth)
    if not truth_words:
        return 0.0
    overlap = answer_words & truth_words
    return len(overlap) / len(truth_words)


# ── Layer 3: LLM-as-Judge ────────────────────────────────────

def llm_judge_score(
    question: str,
    answer: str,
    context: str = "",
    ground_truth: str = "",
    criterion: str = "faithfulness",
) -> float:
    """
    Use LLM as a judge to score an answer on a 1-5 scale.

    Criteria:
      - faithfulness: Is the answer faithful to the context?
      - relevance: Is the answer relevant to the question?
      - correctness: Is the answer correct vs ground truth?
    """
    prompts = {
        "faithfulness": f"""Score this answer's faithfulness to the provided context on a scale of 1-5.

Context: {context[:1500]}
Answer: {answer[:500]}

1 = Completely unfaithful, makes claims not in context
5 = Perfectly faithful, every claim supported by context

Return ONLY a JSON: {{"score": <1-5>, "reason": "brief reason"}}""",

        "relevance": f"""Score this answer's relevance to the question on a scale of 1-5.

Question: {question}
Answer: {answer[:500]}

1 = Completely irrelevant
5 = Directly and thoroughly answers the question

Return ONLY a JSON: {{"score": <1-5>, "reason": "brief reason"}}""",

        "correctness": f"""Score this answer's correctness compared to the ground truth on a scale of 1-5.

Question: {question}
Answer: {answer[:500]}
Ground Truth: {ground_truth[:500]}

1 = Completely incorrect
5 = Perfectly correct and complete

Return ONLY a JSON: {{"score": <1-5>, "reason": "brief reason"}}""",
    }

    prompt = prompts.get(criterion, prompts["faithfulness"])

    try:
        from src.llm.engine import get_llm
        llm = get_llm()
        result = llm.generate_json(prompt, temperature=0.0)
        score = float(result.get("score", 3))
        return max(1.0, min(5.0, score))
    except Exception as e:
        logger.warning("LLM judge failed: %s", e)
        return 3.0  # neutral default


# ── Layer 4: Memory-Specific Metrics ─────────────────────────

def memory_recall_score(
    question: str,
    answer: str,
    planted_facts: list[str],
) -> float:
    """
    Score how well the answer recalls planted user facts.

    Returns fraction of planted facts that are reflected in the answer.
    """
    if not planted_facts:
        return 0.0

    answer_lower = answer.lower()
    recalled = 0
    for fact in planted_facts:
        # Check if key words from the fact appear in the answer
        fact_words = set(re.findall(r"\w+", fact.lower()))
        # Remove common words
        fact_words -= {"user", "the", "is", "a", "an", "and", "or", "to", "in", "of", "has", "their"}
        if fact_words:
            overlap = sum(1 for w in fact_words if w in answer_lower)
            if overlap / len(fact_words) >= 0.5:
                recalled += 1

    return recalled / len(planted_facts)


def personalisation_score(answer: str, planted_facts: list[str]) -> float:
    """
    Score how personalised the answer is based on user facts.
    Similar to memory_recall but specifically checks for personalisation language.
    """
    personalisation_indicators = [
        "you", "your", "based on your", "since you",
        "as you mentioned", "given your preference",
        "knowing that you", "considering your",
    ]

    answer_lower = answer.lower()
    indicator_count = sum(1 for p in personalisation_indicators if p in answer_lower)
    fact_reflection = memory_recall_score("", answer, planted_facts)

    # Combine: 50% fact recall + 50% personalisation language
    personalisation_indicator = min(1.0, indicator_count / 3.0)
    return 0.5 * fact_reflection + 0.5 * personalisation_indicator


# ── Aggregate ────────────────────────────────────────────────

def compute_all_metrics(
    question: str,
    answer: str,
    context: str = "",
    ground_truth: str = "",
    retrieved_ids: Optional[list[str]] = None,
    relevant_ids: Optional[list[str]] = None,
    planted_facts: Optional[list[str]] = None,
    use_llm_judge: bool = True,
) -> dict:
    """Compute all available metrics for a single test case."""
    metrics = {}

    # Layer 1: Retrieval
    if retrieved_ids and relevant_ids:
        metrics["hit_rate"] = hit_rate(retrieved_ids, relevant_ids)
        metrics["mrr"] = mrr(retrieved_ids, relevant_ids)
        metrics["precision_at_5"] = precision_at_k(retrieved_ids, relevant_ids, 5)
        metrics["recall_at_5"] = recall_at_k(retrieved_ids, relevant_ids, 5)
        metrics["ndcg_at_5"] = ndcg_at_k(retrieved_ids, relevant_ids, 5)

    # Layer 2: Lexical
    if context:
        metrics["groundedness"] = round(groundedness(answer, context), 3)
        metrics["hallucination_rate"] = round(hallucination_rate(answer, context), 3)
        metrics["context_utilisation"] = round(context_utilisation(answer, context), 3)
    metrics["answer_relevance"] = round(answer_relevance(answer, question), 3)
    if ground_truth:
        metrics["factual_consistency"] = round(factual_consistency(answer, ground_truth), 3)

    # Layer 3: LLM Judge
    if use_llm_judge and context:
        metrics["judge_faithfulness"] = llm_judge_score(question, answer, context, criterion="faithfulness")
        metrics["judge_relevance"] = llm_judge_score(question, answer, criterion="relevance")
        if ground_truth:
            metrics["judge_correctness"] = llm_judge_score(question, answer, ground_truth=ground_truth, criterion="correctness")

    # Layer 4: Memory
    if planted_facts:
        metrics["memory_recall"] = round(memory_recall_score(question, answer, planted_facts), 3)
        metrics["personalisation"] = round(personalisation_score(answer, planted_facts), 3)

    return metrics
