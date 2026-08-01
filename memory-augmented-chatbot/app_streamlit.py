"""
Streamlit UI — premium multi-panel interface for the chatbot.

Usage:
    streamlit run app_streamlit.py
"""

from __future__ import annotations

import json
import os
import time
import logging

import streamlit as st

# Page config
st.set_page_config(
    page_title="🧠 Memory-Augmented Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #E8E4FB 0%, #DCE4FB 45%, #D4E8FC 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #7C6FEF 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0;
        padding-bottom: 10px;
    }
    .sub-header {
        color: #6B6B93;
        font-size: 1.1em;
        margin-top: -20px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F2F0FE 100%);
        border: 1px solid #D9D4FA;
        border-radius: 14px;
        padding: 16px;
        margin: 4px 0;
        box-shadow: 0 4px 14px rgba(124, 111, 239, 0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #7C6FEF;
        box-shadow: 0 8px 20px rgba(124, 111, 239, 0.18);
    }
    .stButton>button {
        background: linear-gradient(135deg, #7C6FEF 0%, #4FACFE 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(124, 111, 239, 0.35) !important;
    }
    .route-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid #E1DCFA;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(124, 111, 239, 0.08);
    }
    div[data-testid="stChatMessage"]:nth-child(even) {
        background: rgba(240, 244, 254, 0.75);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E6E1FB 0%, #DCE6FC 100%);
        border-right: 1px solid #D2CDF7;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255, 255, 255, 0.6);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #D9D4FA;
        border-bottom: 1px solid #D9D4FA !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        color: #5C5C7A;
        white-space: nowrap;
        min-width: max-content;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"] p {
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(124, 111, 239, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7C6FEF 0%, #4FACFE 100%) !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(124, 111, 239, 0.3);
    }
    /* Kill the default red underline indicator bar */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    .stTabs div[class*="highlight"] {
        background-color: transparent !important;
        box-shadow: none !important;
        display: none !important;
    }

    /* ── Hero banner ── */
    .hero-banner {
        background: rgba(255, 255, 255, 0.55);
        border: 1px solid #D9D4FA;
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
        backdrop-filter: blur(6px);
        box-shadow: 0 8px 24px rgba(124, 111, 239, 0.1);
    }

    /* ── Feature pills ── */
    .feature-pill {
        display: inline-block;
        background: linear-gradient(135deg, rgba(124,111,239,0.12), rgba(79,172,254,0.12));
        border: 1px solid #D2CDF7;
        color: #5A4FCF;
        padding: 5px 14px;
        margin: 4px 6px 0 0;
        border-radius: 999px;
        font-size: 0.82em;
        font-weight: 600;
    }

    /* ── Chat input ── */
    div[data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1px solid #D2CDF7 !important;
        box-shadow: 0 4px 16px rgba(124, 111, 239, 0.12) !important;
        background: #FFFFFF !important;
    }
    div[data-testid="stChatInput"]:focus-within {
        border: 1.5px solid #7C6FEF !important;
        box-shadow: 0 6px 20px rgba(124, 111, 239, 0.25) !important;
    }

    /* ── Suggestion chips ── */
    .stButton>button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #5A4FCF !important;
        border: 1px solid #D9D4FA !important;
        border-radius: 999px !important;
        font-weight: 500 !important;
        padding: 6px 16px !important;
        box-shadow: none !important;
    }
    .stButton>button[kind="secondary"]:hover {
        background: rgba(124, 111, 239, 0.08) !important;
        border-color: #7C6FEF !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


def init_session():
    """Initialise session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default"


def render_sidebar():
    """Render the sidebar with controls and status."""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # User ID
        user_id = st.text_input("👤 User ID", value=st.session_state.user_id)
        st.session_state.user_id = user_id

        st.divider()

        # Provider Status
        st.markdown("### 🔌 LLM Providers")
        try:
            from src.llm.engine import get_llm
            llm = get_llm()
            for p in llm.provider_status():
                icon = "🟢" if p["available"] else "🔴"
                st.markdown(f"{icon} **{p['name'].capitalize()}**")
        except Exception:
            st.warning("LLM engine not initialised")

        st.divider()

        # Knowledge Base Stats
        st.markdown("### 📚 Knowledge Base")
        try:
            from src.rag.vector_store import get_vector_store
            from src.graph.store import get_graph_store
            vs = get_vector_store()
            gs = get_graph_store()
            vs_stats = vs.stats()
            gs_stats = gs.get_stats()
            col1, col2 = st.columns(2)
            col1.metric("📄 Chunks", vs_stats.get("document_count", 0))
            col2.metric("🔗 KG Nodes", gs_stats.get("nodes", 0))
            st.metric("🔗 KG Edges", gs_stats.get("edges", 0))
        except Exception:
            st.info("No data ingested yet")

        st.divider()

        # Pipeline Controls
        st.markdown("### 🔧 Pipeline Controls")

        if st.button("🌐 Ingest from urls.txt", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                try:
                    from src.data.scraper import scrape_urls, load_urls_from_file
                    from src.data.cleaner import clean_all
                    from src.data.chunker import chunk_directory
                    from src.rag.vector_store import get_vector_store
                    from src.config import BASE_DIR

                    urls = load_urls_from_file(BASE_DIR / "urls.txt")
                    scraped = scrape_urls(urls)
                    cleaned = clean_all()
                    chunks = chunk_directory()
                    store = get_vector_store()
                    store.add_chunks(chunks)
                    st.success(f"✅ Ingested {len(chunks)} chunks from {len(scraped)} pages!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

        if st.button("🗑️ Clear Memory", use_container_width=True):
            from src.memory.manager import clear_user_memory
            clear_user_memory(st.session_state.user_id)
            st.success("Memory cleared!")

        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat():
    """Render the main chat panel."""
    st.markdown("""
        <div class="hero-banner">
            <p class="main-header">🧠 Memory-Augmented Chatbot</p>
            <p class="sub-header">An AI that remembers you, reasons over a knowledge graph, and fetches live data.</p>
            <div>
                <span class="feature-pill">🧬 Knowledge Graph</span>
                <span class="feature-pill">🔍 Hybrid RAG</span>
                <span class="feature-pill">💾 Persistent Memory</span>
                <span class="feature-pill">🛠️ 12 Dynamic Tools</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                meta = msg["metadata"]
                cols = st.columns(4)
                route = meta.get("route", "")
                route_color = {
                    "rag": "blue", "kg": "green", "tool": "orange",
                    "direct": "violet", "hybrid": "rainbow",
                }.get(route, "gray")
                cols[0].markdown(f"Route: :{route_color}[{route}]")

                conf = meta.get("confidence", 0)
                conf_color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
                cols[1].markdown(f"Confidence: :{conf_color}[{conf:.0%}]")

                cols[2].markdown(f"Provider: `{meta.get('provider', 'N/A')}`")

                latency = meta.get("latency", {})
                total_ms = sum(latency.values()) if latency else 0
                cols[3].markdown(f"Latency: `{total_ms:.0f}ms`")

    # Interactive suggestion chips — only show when the conversation is empty
    pending_prompt = None
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        suggestions = [
            "What is machine learning?",
            "Explain knowledge graphs",
            "What's the latest AI news?",
        ]
        chip_cols = st.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if chip_cols[i].button(s, key=f"chip_{i}", type="secondary", use_container_width=True):
                pending_prompt = s

    # Chat input
    typed_prompt = st.chat_input("Ask me anything...")
    prompt = typed_prompt or pending_prompt

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.agent.graph import chat
                    result = chat(
                        user_id=st.session_state.user_id,
                        message=prompt,
                    )
                    answer = result.get("answer", "I couldn't generate a response.")
                    st.markdown(answer)

                    # Show metadata
                    meta = {
                        "route": result.get("route", ""),
                        "confidence": result.get("confidence", 0),
                        "provider": result.get("provider", ""),
                        "latency": result.get("latency", {}),
                        "sources": result.get("sources", []),
                    }

                    cols = st.columns(4)
                    route = meta["route"]
                    route_color = {
                        "rag": "blue", "kg": "green", "tool": "orange",
                        "direct": "violet", "hybrid": "rainbow",
                    }.get(route, "gray")
                    cols[0].markdown(f"Route: :{route_color}[{route}]")

                    conf = meta["confidence"]
                    conf_color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
                    cols[1].markdown(f"Confidence: :{conf_color}[{conf:.0%}]")
                    cols[2].markdown(f"Provider: `{meta['provider']}`")
                    total_ms = sum(meta["latency"].values()) if meta["latency"] else 0
                    cols[3].markdown(f"Latency: `{total_ms:.0f}ms`")

                    # Sources expander
                    if meta["sources"]:
                        with st.expander("📚 Sources"):
                            for src in meta["sources"]:
                                st.markdown(f"- `{src}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": meta,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")


def render_memory_tab():
    """Render the Memory Inspector tab."""
    st.markdown("### 🧠 Memory Inspector")

    try:
        from src.memory.manager import get_user_facts
        facts = get_user_facts(st.session_state.user_id)

        if not facts:
            st.info("No memories stored yet. Chat with the bot and share some preferences!")
            return

        st.metric("Total Active Facts", len(facts))

        for fact in facts:
            col1, col2, col3 = st.columns([6, 2, 1])
            col1.markdown(f"**{fact['fact']}**")
            col2.markdown(f"`{fact.get('category', 'general')}`")
            if col3.button("🗑️", key=f"del_{fact['id']}"):
                from src.memory.manager import delete_fact
                delete_fact(fact["id"])
                st.rerun()
    except Exception as e:
        st.error(f"Error loading memory: {e}")


def render_kg_tab():
    """Render the Knowledge Graph Viewer tab."""
    st.markdown("### 🕸️ Knowledge Graph Viewer")

    try:
        from src.graph.store import get_graph_store, NetworkXGraphStore
        graph = get_graph_store()
        stats = graph.get_stats()

        col1, col2, col3 = st.columns(3)
        col1.metric("Nodes", stats.get("nodes", 0))
        col2.metric("Edges", stats.get("edges", 0))
        col3.metric("Backend", stats.get("backend", "N/A"))

        if stats.get("nodes", 0) == 0:
            st.info("Knowledge graph is empty. Run the ingestion pipeline first!")
            return

        # Search
        query = st.text_input("🔍 Search entity", placeholder="e.g., Deep Learning")
        if query:
            results = graph.search_entity(query, limit=10)
            relations = graph.get_entity_relations(query, hops=1)

            if results:
                st.markdown("**Matching entities:**")
                for r in results:
                    st.markdown(f"- **{r['name']}** ({', '.join(r.get('types', []))})")

            if relations:
                st.markdown("**Relationships:**")
                for rel in relations:
                    st.markdown(f"- `{rel}`")

        # Visualisation (NetworkX backend only)
        if isinstance(graph, NetworkXGraphStore) and stats.get("nodes", 0) > 0:
            st.markdown("---")
            st.markdown("**Interactive Graph**")
            try:
                from pyvis.network import Network
                import streamlit.components.v1 as components

                vis_data = graph.get_visualization_data()
                net = Network(height="500px", width="100%", bgcolor="#0e1117", font_color="white")

                type_colors = {
                    "CONCEPT": "#667eea", "PERSON": "#f44336",
                    "ORG": "#ff9800", "TECHNOLOGY": "#4caf50",
                    "ALGORITHM": "#e91e63", "DATASET": "#00bcd4",
                    "METRIC": "#9c27b0",
                }

                for node in vis_data["nodes"]:
                    color = type_colors.get(node.get("group", "CONCEPT"), "#667eea")
                    net.add_node(node["id"], label=node["label"], color=color, title=node.get("title", ""))

                for edge in vis_data["edges"]:
                    net.add_edge(edge["from"], edge["to"], label=edge.get("label", ""), title=edge.get("title", ""))

                net.set_options("""
                {
                    "physics": {
                        "barnesHut": {"gravitationalConstant": -2000, "springLength": 200}
                    }
                }
                """)

                import tempfile
                html_path = os.path.join(tempfile.gettempdir(), "kg_graph.html")
                net.save_graph(html_path)
                with open(html_path, "r") as f:
                    html = f.read()
                components.html(html, height=550)
            except Exception as e:
                st.warning(f"Visualisation unavailable: {e}")

        # Top entities
        if stats.get("top_entities"):
            st.markdown("---")
            st.markdown("**Top Entities by Connections**")
            for e in stats["top_entities"][:10]:
                st.markdown(f"- **{e['name']}**: {e['connections']} connections")

    except Exception as e:
        st.error(f"Error: {e}")


def render_eval_tab():
    """Render the Evaluation Dashboard tab."""
    st.markdown("### 📊 Evaluation Dashboard")

    if st.button("▶️ Run Evaluation", use_container_width=True):
        with st.spinner("Running evaluation... This may take a few minutes."):
            try:
                from src.eval.evaluator import RAGEvaluator
                evaluator = RAGEvaluator()
                summary = evaluator.run_evaluation(use_llm_judge=False)
                evaluator.save_report()

                st.success("Evaluation complete!")
                st.code(evaluator.print_scoreboard())

                # Score cards
                overall = summary.get("overall", {})
                if overall:
                    cols = st.columns(4)
                    cols[0].metric("Groundedness", f"{overall.get('groundedness', 0):.1%}")
                    cols[1].metric("Answer Relevance", f"{overall.get('answer_relevance', 0):.1%}")
                    cols[2].metric("Avg Confidence", f"{overall.get('avg_confidence', 0):.1%}")
                    cols[3].metric("Avg Latency", f"{overall.get('avg_latency_ms', 0):.0f}ms")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    # Show existing results
    from pathlib import Path
    report_path = Path("data/eval_report.json")
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            summary = report.get("summary", {})

            st.markdown("---")
            st.markdown("**Latest Results**")
            st.json(summary)
        except Exception:
            pass


def render_observability_tab():
    """Render the Observability tab."""
    st.markdown("### 📈 Observability")

    try:
        from src.llm.engine import get_llm
        llm = get_llm()
        usage = llm.usage.summary()

        if not usage.get("calls_per_provider"):
            st.info("No usage data yet. Start chatting to generate telemetry!")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Calls per Provider**")
            import plotly.express as px
            calls = usage["calls_per_provider"]
            if calls:
                fig = px.pie(
                    values=list(calls.values()),
                    names=list(calls.keys()),
                    title="API Calls Distribution",
                    color_discrete_sequence=["#667eea", "#764ba2", "#f44336"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Average Latency (ms)**")
            latency = usage.get("avg_latency_ms", {})
            if latency:
                fig = px.bar(
                    x=list(latency.keys()),
                    y=list(latency.values()),
                    title="Avg Latency per Provider",
                    color_discrete_sequence=["#667eea"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    xaxis_title="Provider",
                    yaxis_title="Latency (ms)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Token usage
        tokens = usage.get("tokens_per_provider", {})
        if tokens:
            st.markdown("**Token Usage per Provider**")
            for provider, count in tokens.items():
                st.markdown(f"- **{provider}**: {count:,} tokens")

    except Exception as e:
        st.error(f"Error: {e}")


# ── Main Layout ──────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat", "🧠 Memory", "🕸️ Knowledge Graph", "📊 Evaluation", "📈 Observability"
    ])

    with tab1:
        render_chat()
    with tab2:
        render_memory_tab()
    with tab3:
        render_kg_tab()
    with tab4:
        render_eval_tab()
    with tab5:
        render_observability_tab()


if __name__ == "__main__":
    main()