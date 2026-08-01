"""
RAG Evaluator — runs the full evaluation pipeline.

Supports:
  - Per-category evaluation (RAG, KG, tool, memory, multi-hop)
  - Memory A/B testing (memory vs. no-memory baseline)
  - Latency benchmarking per component
  - JSON report generation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR
from src.eval.metrics import compute_all_metrics
from src.eval.test_cases import get_test_cases, get_memory_test_cases

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """End-to-end evaluator for the chatbot system."""

    def __init__(self):
        self.results: list[dict] = []
        self.summary: dict = {}

    def evaluate_single(
        self,
        question: str,
        ground_truth: str = "",
        category: str = "",
        planted_facts: Optional[list[str]] = None,
        use_llm_judge: bool = True,
    ) -> dict:
        """Evaluate a single question through the full pipeline."""
        from src.agent.graph import chat

        t0 = time.time()
        response = chat(user_id="eval_user", message=question)
        total_latency = (time.time() - t0) * 1000

        answer = response.get("answer", "")
        route = response.get("route", "")

        # Compute metrics
        metrics = compute_all_metrics(
            question=question,
            answer=answer,
            context="",  # We don't have ground-truth context for all cases
            ground_truth=ground_truth,
            planted_facts=planted_facts,
            use_llm_judge=use_llm_judge,
        )

        result = {
            "question": question,
            "answer": answer[:200],
            "ground_truth": ground_truth[:200],
            "category": category,
            "route": route,
            "provider": response.get("provider", ""),
            "confidence": response.get("confidence", 0),
            "total_latency_ms": round(total_latency, 1),
            "component_latency": response.get("latency", {}),
            "metrics": metrics,
        }

        self.results.append(result)
        return result

    def run_evaluation(
        self,
        categories: Optional[list[str]] = None,
        use_llm_judge: bool = True,
    ) -> dict:
        """
        Run evaluation across all test cases.

        Args:
            categories: Filter to specific categories (None = all).
            use_llm_judge: Whether to use LLM-as-judge metrics.

        Returns:
            Summary dict with aggregated metrics.
        """
        test_cases = get_test_cases()
        if categories:
            test_cases = [tc for tc in test_cases if tc["category"] in categories]

        logger.info("Running evaluation on %d test cases...", len(test_cases))
        self.results = []

        for i, tc in enumerate(test_cases):
            logger.info(
                "  [%d/%d] %s: %s",
                i + 1, len(test_cases), tc["category"], tc["question"][:50],
            )

            # Plant facts for memory test cases
            if tc.get("planted_facts"):
                self._plant_facts("eval_user", tc["planted_facts"])

            result = self.evaluate_single(
                question=tc["question"],
                ground_truth=tc.get("ground_truth", ""),
                category=tc["category"],
                planted_facts=tc.get("planted_facts"),
                use_llm_judge=use_llm_judge,
            )

            logger.info(
                "    → route=%s, confidence=%.2f, latency=%.0fms",
                result["route"], result["confidence"], result["total_latency_ms"],
            )

        # Aggregate results
        self.summary = self._aggregate()
        return self.summary

    def run_memory_ab_test(self) -> dict:
        """
        Memory A/B test: compare memory-enabled vs. memoryless responses.

        Protocol:
          1. Plant facts for test user
          2. Ask memory questions WITH memory
          3. Ask same questions WITHOUT memory (fresh user)
          4. Compare scores
        """
        memory_cases = get_memory_test_cases()
        if not memory_cases:
            return {"error": "No memory test cases available"}

        results = {"with_memory": [], "without_memory": []}

        for tc in memory_cases:
            # WITH memory
            self._plant_facts("memory_test_user", tc["planted_facts"])
            from src.agent.graph import chat as agent_chat

            response_with = agent_chat(user_id="memory_test_user", message=tc["question"])
            metrics_with = compute_all_metrics(
                question=tc["question"],
                answer=response_with.get("answer", ""),
                planted_facts=tc["planted_facts"],
                use_llm_judge=False,
            )
            results["with_memory"].append(metrics_with)

            # WITHOUT memory (fresh user, no planted facts)
            response_without = agent_chat(user_id="no_memory_baseline", message=tc["question"])
            metrics_without = compute_all_metrics(
                question=tc["question"],
                answer=response_without.get("answer", ""),
                planted_facts=tc["planted_facts"],
                use_llm_judge=False,
            )
            results["without_memory"].append(metrics_without)

        # Compute average delta
        avg_with = self._avg_metric(results["with_memory"], "memory_recall")
        avg_without = self._avg_metric(results["without_memory"], "memory_recall")

        avg_pers_with = self._avg_metric(results["with_memory"], "personalisation")
        avg_pers_without = self._avg_metric(results["without_memory"], "personalisation")

        return {
            "memory_recall": {
                "with_memory": round(avg_with, 3),
                "without_memory": round(avg_without, 3),
                "uplift": round((avg_with - avg_without) * 100, 1),
            },
            "personalisation": {
                "with_memory": round(avg_pers_with, 3),
                "without_memory": round(avg_pers_without, 3),
                "uplift": round((avg_pers_with - avg_pers_without) * 100, 1),
            },
            "test_count": len(memory_cases),
        }

    def _plant_facts(self, user_id: str, facts: list[str]) -> None:
        """Plant facts into user memory for testing."""
        from src.memory.store import get_memory_store
        store = get_memory_store()
        for fact in facts:
            store.add_fact(user_id, fact, category="test", confidence=0.95)

    def _avg_metric(self, results: list[dict], metric: str) -> float:
        """Average a specific metric across results."""
        values = [r.get(metric, 0.0) for r in results if metric in r]
        return sum(values) / len(values) if values else 0.0

    def _aggregate(self) -> dict:
        """Aggregate results into a summary."""
        if not self.results:
            return {}

        # Per-category averages
        categories: dict[str, list[dict]] = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        summary = {
            "total_cases": len(self.results),
            "categories": {},
            "overall": {},
        }

        # Overall metrics
        all_metrics_keys = set()
        for r in self.results:
            all_metrics_keys.update(r["metrics"].keys())

        for key in all_metrics_keys:
            values = [r["metrics"][key] for r in self.results if key in r["metrics"]]
            if values:
                summary["overall"][key] = round(sum(values) / len(values), 3)

        # Average latency
        latencies = [r["total_latency_ms"] for r in self.results]
        summary["overall"]["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        # Average confidence
        confidences = [r["confidence"] for r in self.results]
        summary["overall"]["avg_confidence"] = round(sum(confidences) / len(confidences), 3)

        # Per-category breakdown
        for cat, cat_results in categories.items():
            cat_summary = {"count": len(cat_results)}
            for key in all_metrics_keys:
                values = [r["metrics"][key] for r in cat_results if key in r["metrics"]]
                if values:
                    cat_summary[key] = round(sum(values) / len(values), 3)

            cat_latencies = [r["total_latency_ms"] for r in cat_results]
            cat_summary["avg_latency_ms"] = round(sum(cat_latencies) / len(cat_latencies), 1)
            summary["categories"][cat] = cat_summary

        return summary

    def save_report(self, filepath: Optional[str] = None) -> str:
        """Save evaluation results as a JSON report."""
        filepath = filepath or str(DATA_DIR / "eval_report.json")
        report = {
            "summary": self.summary,
            "results": self.results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        Path(filepath).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Evaluation report saved to %s", filepath)
        return filepath

    def print_scoreboard(self) -> str:
        """Return a formatted scoreboard string."""
        if not self.summary:
            return "No evaluation results available."

        lines = []
        lines.append("=" * 60)
        lines.append("  📊 EVALUATION SCOREBOARD")
        lines.append("=" * 60)
        lines.append(f"  Total test cases: {self.summary.get('total_cases', 0)}")
        lines.append("")

        overall = self.summary.get("overall", {})
        lines.append("  ── Overall Metrics ──")
        for key, val in sorted(overall.items()):
            lines.append(f"    {key:.<35} {val}")

        lines.append("")
        lines.append("  ── Per-Category Breakdown ──")
        for cat, cat_data in self.summary.get("categories", {}).items():
            lines.append(f"\n    [{cat.upper()}] ({cat_data.get('count', 0)} cases)")
            for key, val in sorted(cat_data.items()):
                if key != "count":
                    lines.append(f"      {key:.<33} {val}")

        lines.append("=" * 60)
        return "\n".join(lines)
