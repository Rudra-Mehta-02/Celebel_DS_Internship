"""
Evaluation runner — executes the full evaluation suite.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --no-llm-judge
    python scripts/run_eval.py --categories rag memory
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM-as-judge metrics")
    parser.add_argument("--categories", nargs="+", help="Filter categories (rag, kg, tool, memory, multi_hop)")
    parser.add_argument("--ab-test", action="store_true", help="Run Memory A/B test")

    args = parser.parse_args()

    from src.eval.evaluator import RAGEvaluator

    evaluator = RAGEvaluator()

    if args.ab_test:
        print("\n🧪 Running Memory A/B Test...\n")
        ab_results = evaluator.run_memory_ab_test()
        print("=" * 50)
        print("  Memory A/B Test Results")
        print("=" * 50)
        for metric, data in ab_results.items():
            if isinstance(data, dict):
                print(f"\n  {metric}:")
                print(f"    With memory:    {data.get('with_memory', 0):.3f}")
                print(f"    Without memory: {data.get('without_memory', 0):.3f}")
                print(f"    Uplift:         {data.get('uplift', 0):+.1f}%")
        print("=" * 50)
    else:
        print("\n📊 Running Evaluation Suite...\n")
        summary = evaluator.run_evaluation(
            categories=args.categories,
            use_llm_judge=not args.no_llm_judge,
        )

        # Print scoreboard
        print(evaluator.print_scoreboard())

        # Save report
        report_path = evaluator.save_report()
        print(f"\n📄 Report saved to: {report_path}")


if __name__ == "__main__":
    main()
