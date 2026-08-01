"""
FastAPI application — 18 REST endpoints for the chatbot system.

Usage:
    uvicorn app:app --reload --port 8000
    Then visit: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(
    title="Memory-Augmented Chatbot API",
    description=(
        "A chatbot with Knowledge Graph, Hybrid RAG (Dense + BM25 + RRF), "
        "persistent user memory, 12 dynamic tools, and multi-provider LLM engine."
    ),
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: list[str]
    confidence: float
    latency: dict
    provider: str
    rewritten_query: str


class ScrapeRequest(BaseModel):
    urls: list[str]
    skip_kg: bool = False


class IngestResponse(BaseModel):
    pages_scraped: int
    pages_cleaned: int
    chunks_created: int
    entities_extracted: int
    message: str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check and system info."""
    return {
        "status": "running",
        "name": "Memory-Augmented Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check — DB connections, model status, providers."""
    from src.config import get_settings
    from src.llm.engine import get_llm
    from src.rag.vector_store import get_vector_store
    from src.graph.store import get_graph_store

    settings = get_settings()
    llm = get_llm()

    return {
        "status": "healthy",
        "providers": llm.provider_status(),
        "vector_store": get_vector_store().stats(),
        "graph_store": get_graph_store().get_stats(),
        "config": {
            "has_groq": settings.has_groq,
            "has_gemini": settings.has_gemini,
            "has_neo4j": settings.has_neo4j,
            "has_postgres": settings.has_postgres,
        },
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """Main chat endpoint — processes a message through the full agent pipeline."""
    from src.agent.graph import chat as agent_chat

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = agent_chat(user_id=request.user_id, message=request.message)
    return ChatResponse(**result)


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint — returns tokens via Server-Sent Events (SSE)."""
    from src.agent.graph import chat as agent_chat
    import json

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    def generate():
        result = agent_chat(user_id=request.user_id, message=request.message)
        # Simulate streaming by yielding word by word
        words = result.get("answer", "").split()
        for i, word in enumerate(words):
            data = json.dumps({"token": word + " ", "done": i == len(words) - 1})
            yield f"data: {data}\n\n"
        # Final metadata
        meta = json.dumps({
            "route": result.get("route"),
            "confidence": result.get("confidence"),
            "sources": result.get("sources", []),
            "provider": result.get("provider"),
            "done": True,
        })
        yield f"data: {meta}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/memory/{user_id}", tags=["Memory"])
def get_memory(user_id: str):
    """Get all stored user facts."""
    from src.memory.manager import get_user_facts
    facts = get_user_facts(user_id)
    return {"user_id": user_id, "facts": facts, "count": len(facts)}


@app.delete("/memory/{user_id}", tags=["Memory"])
def clear_memory(user_id: str):
    """Clear all user memory."""
    from src.memory.manager import clear_user_memory
    clear_user_memory(user_id)
    return {"message": f"Memory cleared for user {user_id}"}


@app.delete("/memory/{user_id}/{fact_id}", tags=["Memory"])
def delete_fact(user_id: str, fact_id: str):
    """Delete a specific fact."""
    from src.memory.manager import delete_fact as del_fact
    del_fact(fact_id)
    return {"message": f"Fact {fact_id} deleted"}


@app.get("/chat/history/{user_id}", tags=["Chat"])
def get_history(user_id: str, limit: int = 20):
    """Get chat history for a user."""
    from src.memory.store import get_memory_store
    store = get_memory_store()
    history = store.get_history(user_id, limit=limit)
    return {"user_id": user_id, "history": history, "count": len(history)}


@app.get("/kg/stats", tags=["Knowledge Graph"])
def kg_stats():
    """Get knowledge graph statistics."""
    from src.graph.store import get_graph_store
    return get_graph_store().get_stats()


@app.get("/kg/entity/{name}", tags=["Knowledge Graph"])
def kg_entity(name: str, hops: int = 1):
    """Look up an entity and its relationships."""
    from src.graph.store import get_graph_store
    graph = get_graph_store()
    entities = graph.search_entity(name)
    relations = graph.get_entity_relations(name, hops=hops)
    return {"entities": entities, "relations": relations}


@app.get("/kg/search", tags=["Knowledge Graph"])
def kg_search(q: str, limit: int = 10):
    """Fuzzy entity search."""
    from src.graph.store import get_graph_store
    results = get_graph_store().search_entity(q, limit=limit)
    return {"query": q, "results": results}


@app.post("/data/scrape", tags=["Data Pipeline"])
def scrape_data(request: ScrapeRequest):
    """Trigger web scraping from a list of URLs."""
    from src.data.scraper import scrape_urls
    from src.data.cleaner import clean_all
    from src.data.chunker import chunk_directory
    from src.rag.vector_store import get_vector_store
    from src.graph.extractor import extract_from_chunks
    from src.graph.store import get_graph_store

    # Scrape
    scraped = scrape_urls(request.urls)

    # Clean
    cleaned = clean_all()

    # Chunk
    chunks = chunk_directory()

    # Add to vector store
    store = get_vector_store()
    store.add_chunks(chunks)

    # Extract KG (unless skipped)
    entities_count = 0
    if not request.skip_kg:
        chunk_dicts = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        extractions = extract_from_chunks(chunk_dicts, sample_rate=0.3)
        graph = get_graph_store()
        for ext in extractions:
            graph.add_extraction(ext)
            entities_count += len(ext.get("entities", []))

    return IngestResponse(
        pages_scraped=len(scraped),
        pages_cleaned=len(cleaned),
        chunks_created=len(chunks),
        entities_extracted=entities_count,
        message="Ingestion pipeline complete",
    )


@app.post("/data/ingest", tags=["Data Pipeline"])
def ingest_from_file():
    """Run the full ingestion pipeline from urls.txt."""
    from src.data.scraper import scrape_urls, load_urls_from_file
    from src.data.cleaner import clean_all
    from src.data.chunker import chunk_directory
    from src.rag.vector_store import get_vector_store
    from src.graph.extractor import extract_from_chunks
    from src.graph.store import get_graph_store
    from src.config import BASE_DIR

    url_file = BASE_DIR / "urls.txt"
    if not url_file.exists():
        raise HTTPException(status_code=404, detail="urls.txt not found")

    urls = load_urls_from_file(url_file)
    scraped = scrape_urls(urls)
    cleaned = clean_all()
    chunks = chunk_directory()

    store = get_vector_store()
    store.add_chunks(chunks)

    chunk_dicts = [{"text": c.text, "metadata": c.metadata} for c in chunks]
    extractions = extract_from_chunks(chunk_dicts, sample_rate=0.3)
    graph = get_graph_store()
    entities_count = 0
    for ext in extractions:
        graph.add_extraction(ext)
        entities_count += len(ext.get("entities", []))

    return IngestResponse(
        pages_scraped=len(scraped),
        pages_cleaned=len(cleaned),
        chunks_created=len(chunks),
        entities_extracted=entities_count,
        message="Full ingestion pipeline complete",
    )


@app.post("/data/upload", tags=["Data Pipeline"])
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file to the knowledge base."""
    from src.data.loader import load_pdf, load_txt
    from src.data.chunker import chunk_text
    from src.rag.vector_store import get_vector_store
    from src.config import DATA_DIR

    # Save uploaded file
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    filepath = upload_dir / file.filename
    content = await file.read()
    filepath.write_bytes(content)

    # Load based on type
    if file.filename.lower().endswith(".pdf"):
        doc = load_pdf(filepath)
    else:
        doc = load_txt(filepath)

    if not doc:
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    # Chunk and index
    chunks = chunk_text(doc["text"], source=file.filename, title=doc.get("title", ""))
    store = get_vector_store()
    added = store.add_chunks(chunks)

    return {
        "filename": file.filename,
        "chunks_created": added,
        "title": doc.get("title", ""),
        "message": "Document uploaded and indexed",
    }


@app.get("/data/status", tags=["Data Pipeline"])
def data_status():
    """Get pipeline status and statistics."""
    from src.rag.vector_store import get_vector_store
    from src.graph.store import get_graph_store

    return {
        "vector_store": get_vector_store().stats(),
        "knowledge_graph": get_graph_store().get_stats(),
    }


@app.post("/rag/retrieve", tags=["RAG"])
def rag_retrieve(query: str, top_k: int = 5):
    """Retrieval-only — no generation."""
    from src.rag.vector_store import get_vector_store
    results = get_vector_store().hybrid_search(query, top_k=top_k)
    return {"query": query, "results": results}


@app.post("/eval/run", tags=["Evaluation"])
def run_evaluation(use_llm_judge: bool = True):
    """Run the full evaluation suite."""
    from src.eval.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    summary = evaluator.run_evaluation(use_llm_judge=use_llm_judge)
    evaluator.save_report()
    return {
        "summary": summary,
        "scoreboard": evaluator.print_scoreboard(),
    }


@app.get("/eval/results", tags=["Evaluation"])
def get_eval_results():
    """Get the latest evaluation results."""
    import json
    report_path = Path("data/eval_report.json")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Run /eval/run first.")
    return json.loads(report_path.read_text())
