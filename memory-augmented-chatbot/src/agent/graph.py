"""
LangGraph workflow — 9-node agent with self-reflection.

Nodes:
  1. memory_node     — load user facts + chat history
  2. rewrite_node    — rewrite query using conversation context
  3. router_node     — classify query → route
  4. rag_node        — hybrid retrieval (dense + BM25 + RRF)
  5. kg_node         — knowledge graph entity lookup
  6. tool_node       — execute dynamic tool
  7. answer_node     — generate response with full context
  8. reflect_node    — self-reflection (confidence check)
  9. fact_node       — extract and store durable facts

Edge graph:
  START → memory → rewrite → router → {rag→kg, kg, tool, direct} → answer → reflect → fact → END
  reflect →(retry)→ rag  (self-correction loop)
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agent.state import ChatState
from src.llm.engine import get_llm
from src.tools.tools import execute_tool, get_tool_descriptions

logger = logging.getLogger(__name__)


# ── Node Implementations ────────────────────────────────────

def memory_node(state: ChatState) -> ChatState:
    """Load user facts and chat history from the memory store."""
    t0 = time.time()
    from src.memory.manager import get_user_context

    user_id = state.get("user_id", "default")
    context = get_user_context(user_id)

    facts = [f["fact"] for f in context["facts"]]
    history = context["history"]

    latency = state.get("latency", {})
    latency["memory_node"] = (time.time() - t0) * 1000

    return {
        **state,
        "user_facts": facts,
        "chat_history": history,
        "latency": latency,
        "reflection_count": 0,
        "needs_retry": False,
    }


def rewrite_node(state: ChatState) -> ChatState:
    """Rewrite the query using conversation context for better retrieval."""
    t0 = time.time()

    message = state.get("message", "")
    history = state.get("chat_history", [])

    # If no history or message is self-contained, skip rewriting
    if not history or len(message.split()) > 15:
        latency = state.get("latency", {})
        latency["rewrite_node"] = (time.time() - t0) * 1000
        return {**state, "rewritten_query": message, "latency": latency}

    # Build context from last few turns
    recent = history[-4:] if len(history) > 4 else history
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in recent
    )

    prompt = f"""Given this conversation history and the latest user message,
rewrite the user's message as a standalone question that includes all necessary context.
If the message is already clear and standalone, return it unchanged.

Conversation:
{history_text}

Latest message: {message}

Rewritten standalone question:"""

    try:
        llm = get_llm()
        resp = llm.generate_fast(prompt, temperature=0.0, max_tokens=256)
        rewritten = resp.text.strip().strip('"')
        if not rewritten:
            rewritten = message
        logger.debug("Query rewrite: '%s' → '%s'", message, rewritten)
    except Exception:
        rewritten = message

    latency = state.get("latency", {})
    latency["rewrite_node"] = (time.time() - t0) * 1000
    return {**state, "rewritten_query": rewritten, "latency": latency}


def router_node(state: ChatState) -> ChatState:
    """Route the query to the appropriate handler."""
    t0 = time.time()

    query = state.get("rewritten_query", state.get("message", ""))
    tool_descriptions = get_tool_descriptions()

    prompt = f"""You are a query router. Classify the user's query into exactly ONE route.

Available routes:
- "rag": The query asks about factual knowledge that might be in our knowledge base (AI, ML, NLP topics).
- "kg": The query asks about relationships between entities, connections, or structured facts.
- "tool": The query needs real-time data or computation. Available tools:
{tool_descriptions}
- "direct": The query is general conversation, greeting, or can be answered without external data.
- "hybrid": The query needs BOTH knowledge base retrieval AND knowledge graph facts.

If the route is "tool", also specify which tool to use and the argument.

Return ONLY valid JSON:
{{"route": "rag|kg|tool|direct|hybrid", "tool_name": "tool_name_or_null", "tool_args": "args_or_null", "reasoning": "brief reason"}}

User query: {query}"""

    try:
        llm = get_llm()
        result = llm.generate_json(prompt, temperature=0.0, max_tokens=256)
        route = result.get("route", "direct")
        tool_name = result.get("tool_name")
        tool_args = result.get("tool_args")

        # Validate route
        valid_routes = {"rag", "kg", "tool", "direct", "hybrid"}
        if route not in valid_routes:
            route = "direct"

        logger.info("Router: '%s' → %s (reason: %s)", query[:50], route, result.get("reasoning", ""))
    except Exception as e:
        logger.warning("Router failed: %s — defaulting to 'direct'", e)
        route = "direct"
        tool_name = None
        tool_args = None

    latency = state.get("latency", {})
    latency["router_node"] = (time.time() - t0) * 1000
    return {
        **state,
        "route": route,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "latency": latency,
    }


def rag_node(state: ChatState) -> ChatState:
    """Retrieve relevant chunks using hybrid search."""
    t0 = time.time()
    from src.rag.vector_store import get_vector_store

    query = state.get("rewritten_query", state.get("message", ""))
    store = get_vector_store()

    # Use more chunks on retry
    reflection_count = state.get("reflection_count", 0)
    top_k = 5 + (reflection_count * 3)  # 5, 8, 11 on retries

    chunks = store.hybrid_search(query, top_k=top_k)

    latency = state.get("latency", {})
    latency["rag_node"] = (time.time() - t0) * 1000
    return {**state, "rag_context": chunks, "latency": latency}


def kg_node(state: ChatState) -> ChatState:
    """Look up entities and relationships in the knowledge graph."""
    t0 = time.time()
    from src.graph.store import get_graph_store

    query = state.get("rewritten_query", state.get("message", ""))
    graph = get_graph_store()

    # Extract key entity from query using simple heuristics
    # (Skip common words, take longest capitalised phrase)
    words = query.split()
    entity_candidates = []
    for i in range(len(words)):
        for j in range(i + 1, min(i + 5, len(words) + 1)):
            phrase = " ".join(words[i:j])
            # Keep phrases that look like entity names
            if phrase[0].isupper() or len(phrase.split()) >= 2:
                entity_candidates.append(phrase)

    # Also try the full query
    entity_candidates.append(query)

    facts = []
    for candidate in entity_candidates[:3]:  # Try top 3 candidates
        candidate_facts = graph.get_entity_relations(candidate, hops=1)
        facts.extend(candidate_facts)
        if facts:
            break

    # Deduplicate
    facts = list(dict.fromkeys(facts))[:15]

    latency = state.get("latency", {})
    latency["kg_node"] = (time.time() - t0) * 1000
    return {**state, "kg_context": facts, "latency": latency}


def tool_node(state: ChatState) -> ChatState:
    """Execute the selected dynamic tool."""
    t0 = time.time()

    tool_name = state.get("tool_name", "")
    tool_args = state.get("tool_args", "")

    if not tool_name:
        result = "No tool specified."
    else:
        result = execute_tool(tool_name, tool_args)

    latency = state.get("latency", {})
    latency["tool_node"] = (time.time() - t0) * 1000
    return {**state, "tool_result": result, "latency": latency}


def answer_node(state: ChatState) -> ChatState:
    """Generate the final answer using all available context."""
    t0 = time.time()

    user_facts = state.get("user_facts", [])
    chat_history = state.get("chat_history", [])
    rag_context = state.get("rag_context", [])
    kg_context = state.get("kg_context", [])
    tool_result = state.get("tool_result")
    message = state.get("message", "")
    route = state.get("route", "direct")

    # Build the prompt
    sections = []

    if user_facts:
        facts_text = "\n".join(f"  - {f}" for f in user_facts)
        sections.append(f"USER FACTS (use these to personalise your response):\n{facts_text}")

    if chat_history:
        recent = chat_history[-6:]
        history_text = "\n".join(f"  {m['role'].upper()}: {m['content']}" for m in recent)
        sections.append(f"RECENT CONVERSATION:\n{history_text}")

    if rag_context:
        chunks_text = "\n\n".join(
            f"[Source: {c.get('metadata', {}).get('title', 'unknown')}]\n{c['text']}"
            for c in rag_context[:5]
        )
        sections.append(f"RETRIEVED KNOWLEDGE:\n{chunks_text}")

    if kg_context:
        kg_text = "\n".join(f"  - {f}" for f in kg_context)
        sections.append(f"KNOWLEDGE GRAPH FACTS:\n{kg_text}")

    if tool_result:
        sections.append(f"TOOL RESULT ({state.get('tool_name', 'tool')}):\n{tool_result}")

    context = "\n\n---\n\n".join(sections)

    system = """You are a helpful, knowledgeable AI assistant. Answer the user's question using the provided context.

Rules:
1. If user facts are provided, use them to personalise your response.
2. Cite sources when using retrieved knowledge.
3. Be concise but thorough.
4. If you don't have enough information, say so honestly.
5. At the end, rate your confidence in this answer on a scale of 0.0 to 1.0.
   Format: [CONFIDENCE: 0.X]"""

    prompt = f"""{context}

USER QUESTION: {message}

Provide a helpful, personalised answer:"""

    try:
        llm = get_llm()
        resp = llm.generate(prompt, system=system, temperature=0.5, max_tokens=1500)

        answer = resp.text
        provider = resp.provider

        # Extract confidence score
        confidence = 0.7  # default
        import re
        conf_match = re.search(r"\[CONFIDENCE:\s*([\d.]+)\]", answer)
        if conf_match:
            confidence = min(1.0, max(0.0, float(conf_match.group(1))))
            answer = answer[:conf_match.start()].strip()

        # Collect sources
        sources = []
        for c in rag_context[:5]:
            src = c.get("metadata", {}).get("source", "")
            if src and src not in sources:
                sources.append(src)
        if kg_context:
            sources.append("knowledge_graph")
        if tool_result:
            sources.append(f"tool:{state.get('tool_name', 'unknown')}")

    except Exception as e:
        answer = f"I encountered an error generating a response: {e}"
        confidence = 0.0
        sources = []
        provider = "error"

    latency = state.get("latency", {})
    latency["answer_node"] = (time.time() - t0) * 1000
    return {
        **state,
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "provider_used": provider,
        "latency": latency,
    }


def reflect_node(state: ChatState) -> ChatState:
    """Self-reflection: check answer quality, retry if needed."""
    confidence = state.get("confidence", 0.7)
    reflection_count = state.get("reflection_count", 0)
    route = state.get("route", "direct")

    # Only retry for knowledge-based routes, not direct/tool
    if route in ("direct", "tool"):
        return {**state, "needs_retry": False}

    # Retry if confidence is low and we haven't retried too many times
    if confidence < 0.4 and reflection_count < 2:
        logger.info(
            "Self-reflection: confidence=%.2f, retrying (attempt %d)",
            confidence, reflection_count + 1,
        )
        return {
            **state,
            "needs_retry": True,
            "reflection_count": reflection_count + 1,
        }

    return {**state, "needs_retry": False}


def fact_extraction_node(state: ChatState) -> ChatState:
    """Extract and store durable facts from the user's message."""
    t0 = time.time()
    from src.memory.manager import process_message

    user_id = state.get("user_id", "default")
    message = state.get("message", "")
    answer = state.get("answer", "")

    # Store user message and bot response in history
    process_message(user_id, "user", message)
    process_message(user_id, "assistant", answer)

    latency = state.get("latency", {})
    latency["fact_extraction_node"] = (time.time() - t0) * 1000
    return {**state, "latency": latency}


# ── Router Edges ─────────────────────────────────────────────

def route_query(state: ChatState) -> str:
    """Determine which node to go to based on the route."""
    route = state.get("route", "direct")
    if route in ("rag", "hybrid"):
        return "rag"
    elif route == "kg":
        return "kg"
    elif route == "tool":
        return "tool"
    else:
        return "direct"


def should_retry(state: ChatState) -> str:
    """Decide whether to retry or finish."""
    if state.get("needs_retry", False):
        return "retry"
    return "finish"


# ── Build the Graph ──────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("memory", memory_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("kg", kg_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("fact_extraction", fact_extraction_node)

    # Entry point
    graph.set_entry_point("memory")

    # Linear edges
    graph.add_edge("memory", "rewrite")
    graph.add_edge("rewrite", "router")

    # Conditional routing
    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "rag": "rag",
            "kg": "kg",
            "tool": "tool",
            "direct": "answer",
        },
    )

    # RAG → KG (hybrid chain: vector + graph always combined)
    graph.add_edge("rag", "kg")

    # All paths converge at answer
    graph.add_edge("kg", "answer")
    graph.add_edge("tool", "answer")

    # Answer → reflect
    graph.add_edge("answer", "reflect")

    # Reflect → retry or finish
    graph.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "retry": "rag",
            "finish": "fact_extraction",
        },
    )

    # Fact extraction → END
    graph.add_edge("fact_extraction", END)

    return graph


# ── Public API ───────────────────────────────────────────────

_compiled = None


def get_agent():
    """Return the compiled LangGraph agent."""
    global _compiled
    if _compiled is None:
        graph = build_graph()
        _compiled = graph.compile()
    return _compiled


def chat(user_id: str, message: str) -> dict:
    """
    Process a chat message through the full agent pipeline.

    Args:
        user_id: User identifier for memory.
        message: User's message.

    Returns:
        {answer, route, sources, confidence, latency, provider}
    """
    agent = get_agent()

    initial_state: ChatState = {
        "user_id": user_id,
        "message": message,
        "user_facts": [],
        "chat_history": [],
        "rewritten_query": "",
        "route": "",
        "tool_name": None,
        "tool_args": None,
        "rag_context": [],
        "kg_context": [],
        "tool_result": None,
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "reflection_count": 0,
        "needs_retry": False,
        "latency": {},
        "provider_used": "",
        "error": None,
    }

    try:
        result = agent.invoke(initial_state)
        return {
            "answer": result.get("answer", ""),
            "route": result.get("route", "direct"),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "latency": result.get("latency", {}),
            "provider": result.get("provider_used", ""),
            "rewritten_query": result.get("rewritten_query", message),
        }
    except Exception as e:
        logger.error("Agent error: %s", e)
        return {
            "answer": f"I'm sorry, I encountered an error: {e}",
            "route": "error",
            "sources": [],
            "confidence": 0.0,
            "latency": {},
            "provider": "error",
            "rewritten_query": message,
        }
