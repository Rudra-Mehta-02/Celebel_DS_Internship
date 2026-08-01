"""
CLI chat client — rich terminal interface.

Usage:
    python scripts/chat_cli.py
    python scripts/chat_cli.py --user alice

Commands:
    /memory  — Show stored user facts
    /forget  — Clear all memory
    /stats   — Show system stats
    /quit    — Exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="CLI Chat Client")
    parser.add_argument("--user", "-u", default="cli_user", help="User ID")
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print("Install rich: pip install rich")
        return

    console = Console()

    console.print(Panel.fit(
        "[bold magenta]🧠 Memory-Augmented Chatbot[/bold magenta]\n"
        "[dim]Knowledge Graph • Hybrid RAG • Persistent Memory • 12 Tools[/dim]\n\n"
        f"[cyan]User:[/cyan] {args.user}\n"
        "[dim]Commands: /memory, /forget, /stats, /quit[/dim]",
        border_style="bright_magenta",
    ))

    from src.agent.graph import chat

    while True:
        try:
            user_input = console.input("\n[bold cyan]You>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/memory":
            from src.memory.manager import get_user_facts
            facts = get_user_facts(args.user)
            if facts:
                table = Table(title="🧠 User Memory")
                table.add_column("Fact", style="white")
                table.add_column("Category", style="cyan")
                table.add_column("Confidence", style="green")
                for f in facts:
                    table.add_row(f["fact"], f.get("category", ""), f"{f.get('confidence', 0):.0%}")
                console.print(table)
            else:
                console.print("[dim]No memories stored yet.[/dim]")
            continue

        if user_input.lower() == "/forget":
            from src.memory.manager import clear_user_memory
            clear_user_memory(args.user)
            console.print("[green]Memory cleared![/green]")
            continue

        if user_input.lower() == "/stats":
            from src.rag.vector_store import get_vector_store
            from src.graph.store import get_graph_store
            from src.llm.engine import get_llm
            vs = get_vector_store().stats()
            gs = get_graph_store().get_stats()
            llm = get_llm()
            usage = llm.usage.summary()

            table = Table(title="📊 System Stats")
            table.add_column("Component", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("Vector Chunks", str(vs.get("document_count", 0)))
            table.add_row("KG Nodes", str(gs.get("nodes", 0)))
            table.add_row("KG Edges", str(gs.get("edges", 0)))
            table.add_row("LLM Calls", str(sum(usage.get("calls_per_provider", {}).values())))
            console.print(table)
            continue

        # Chat
        with console.status("[bold magenta]Thinking...[/bold magenta]"):
            result = chat(user_id=args.user, message=user_input)

        answer = result.get("answer", "Error")
        route = result.get("route", "unknown")
        confidence = result.get("confidence", 0)
        provider = result.get("provider", "unknown")
        latency = result.get("latency", {})
        total_ms = sum(latency.values()) if latency else 0

        # Display answer
        console.print(f"\n[bold green]Bot>[/bold green]", end=" ")
        console.print(Markdown(answer))

        # Display metadata
        route_colors = {
            "rag": "blue", "kg": "green", "tool": "yellow",
            "direct": "magenta", "hybrid": "cyan",
        }
        route_color = route_colors.get(route, "white")
        conf_color = "green" if confidence > 0.7 else "yellow" if confidence > 0.4 else "red"

        console.print(
            f"  [{route_color}]▸ Route: {route}[/{route_color}]  "
            f"[{conf_color}]▸ Confidence: {confidence:.0%}[/{conf_color}]  "
            f"[dim]▸ Provider: {provider} ▸ Latency: {total_ms:.0f}ms[/dim]"
        )


if __name__ == "__main__":
    main()
