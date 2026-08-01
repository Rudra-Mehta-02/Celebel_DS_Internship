    <div align="center">
    
Memory-Augmented Chatbot with Hybrid RAG & Knowledge Graph
🚀 An intelligent AI assistant that combines Hybrid RAG, Knowledge Graph reasoning, persistent user memory, and dynamic tool calling to deliver personalized, context-aware, and real-time responses through an agentic LangGraph workflow.

| Capability | Implementation |
|---|---|
| 🔍 Hybrid Retrieval | ChromaDB + BM25 |
| 🧬 Knowledge Graph | Neo4j / NetworkX |
| 💾 Long-Term Memory | PostgreSQL / SQLite |
| 🔀 Orchestration | LangGraph (9 Nodes) |
| ⚡ LLM Reliability | Multi-Provider Failover |
| 🔄 Quality Control | Self-Reflection Loop |
| 📊 Testing | Evaluation Framework |

</div>

---

Every request is orchestrated through a **9-node LangGraph agentic workflow** that dynamically selects the optimal execution path—whether querying a vector knowledge base, traversing a knowledge graph, invoking a live API, or responding directly. The system delivers **context-aware, personalized responses** by leveraging durable user memories extracted from previous conversations and stored in a persistent database. To improve reliability, it incorporates a **self-reflection mechanism** that automatically triggers deeper retrieval whenever response confidence falls below a defined threshold.

Designed with **100% free and open-access APIs**, the platform features a resilient **multi-provider LLM failover architecture** (**Groq → Gemini → Ollama**), ensuring uninterrupted operation by seamlessly switching providers when rate limits or service disruptions occur.

🎓 **Engineered as the capstone project for the Celebal Technologies Internship Program (CEIP-2026), showcasing advanced agentic AI workflows, retrieval orchestration, long-term memory, and fault-tolerant LLM infrastructure.**

---

## 📑 Table of Contents

1. [✨ Key Features & Technical Highlights](#1--key-features--technical-highlights)
2. [🏗️ System Architecture](#2--system-architecture)
3. [🚀 End-to-End Request Lifecycle](#3--end-to-end-request-lifecycle)
4. [🧠 Long-Term & Session Memory Architecture](#4--long-term--session-memory-architecture)
5. [🔍 Hybrid Retrieval Pipeline (Dense + BM25 + RRF)](#5--hybrid-retrieval-pipeline-dense--bm25--rrf)
6. [🔀 Agent Workflow (9-Node LangGraph)](#6--agent-workflow-9-node-langgraph)
7. [🛠️ Dynamic Tooling & Function Calling](#7--dynamic-tooling--function-calling)
8. [📊 Performance Evaluation & Benchmarks](#8--performance-evaluation--benchmarks)
9. [🧩 Core Components Explained](#9--core-components-explained)
10. [⚙️ Technology Stack](#10--technology-stack)
11. [🚀 Getting Started](#11--getting-started)
12. [📡 REST API Reference (18 Endpoints)](#12--rest-api-reference-18-endpoints)
13. [📁 Repository Structure](#13--repository-structure)
14. [🎯 Architecture Decisions & Trade-offs](#14--architecture-decisions--trade-offs)
15. [🔭 Limitations & Future Enhancements](#15--limitations--future-enhancements)

---
## 1. ✨ Key Features & Technical Highlights

Unlike a traditional RAG chatbot that simply retrieves relevant documents and generates responses, this system combines **agentic workflows, hybrid retrieval, persistent memory, and resilient infrastructure** to deliver more accurate, personalized, and reliable answers.

### 🏗️ Architecture & Reasoning

| # | Capability               | This Project                                    | Standard RAG               |
| - | ------------------------ | ----------------------------------------------- | -------------------------- |
| 1 | **Multi-Provider LLM**   | Groq → Gemini → Ollama auto-failover            | ❌ Single provider          |
| 2 | **Streaming Responses**  | Real-time SSE token streaming                   | ❌ Waits for full response  |
| 3 | **Self-Reflection Loop** | Retries with deeper retrieval on low confidence | ❌ One-shot generation      |
| 4 | **Query Rewriting**      | Resolves follow-up questions using context      | ❌ Independent queries      |
| 5 | **Confidence Scoring**   | Confidence score (0.0–1.0) for every response   | ❌ No confidence estimation |

### 🔍 Intelligent Retrieval

| # | Capability                    | This Project                                | Standard RAG           |
| - | ----------------------------- | ------------------------------------------- | ---------------------- |
| 6 | **Hybrid Retrieval**          | Dense + BM25 + RRF Fusion                   | ❌ Dense-only retrieval |
| 7 | **Graph-Augmented Retrieval** | Combines vector search with Knowledge Graph | ❌ Separate KG and RAG  |
| 8 | **Semantic Memory**           | Retrieves memories by relevance             | ❌ Recency-based memory |
| 9 | **Adaptive Chunking**         | Context-aware document splitting            | ❌ Fixed-size chunks    |

### 🚀 Production Features

| #  | Capability                  | This Project                        | Standard RAG                      |
| -- | --------------------------- | ----------------------------------- | --------------------------------- |
| 10 | **Observability Dashboard** | Latency, tokens, routing metrics    | ❌ Limited monitoring              |
| 11 | **Knowledge Ingestion**     | Upload PDFs, TXT, and URLs          | ❌ Manual ingestion                |
| 12 | **Memory Versioning**       | Resolves conflicting memories       | ❌ Overwrites or ignores conflicts |
| 13 | **Conversation Summaries**  | Compresses long chats automatically | ❌ Hard truncation                 |
| 14 | **Rate-Limit Awareness**    | Smart provider switching            | ❌ No failover strategy            |
| 15 | **Latency Tracking**        | Per-component performance metrics   | ❌ End-to-end timing only          |


---

## 2. 🏗️ System Architecture

```
                              ┌──────────────────────────────────┐
                              │      PRESENTATION LAYER          │
                              │                                  │
                              │  Streamlit UI ─── FastAPI ─── CLI│
                              │  • Chat + Streaming (SSE)        │
                              │  • KG Visualizer (pyvis)         │
                              │  • Memory Inspector              │
                              │  • Eval Dashboard                │
                              │  • Observability Panel           │
                              │  • Document Upload               │
                              └──────────┬───────────────────────┘
                                         │
                              ┌──────────▼───────────────────────┐
                              │   MULTI-PROVIDER LLM ENGINE      │
                              │                                  │
                              │   Groq ──► Gemini ──► Ollama     │
                              │   (auto-failover + rate tracking)│
                              └──────────┬───────────────────────┘
                                         │
┌────────────────────────────────────────▼──────────────────────────────────────┐
│                        LANGGRAPH ORCHESTRATION (9 NODES)                      │
│                                                                              │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Memory  │─▶│ Rewriter │─▶│  Router   │─▶│ Retrieve │─▶│   Answer     │  │
│  │ Loader  │  │ (query   │  │  (LLM,    │  │ (parallel│  │   Generator  │  │
│  │         │  │  rewrite)│  │  temp=0)  │  │  fusion) │  │              │  │
│  └─────────┘  └──────────┘  └─────┬─────┘  └──────────┘  └──────┬───────┘  │
│                                   │                              │           │
│                        ┌──────────┼──────────┐          ┌───────▼────────┐  │
│                        ▼          ▼          ▼          │ Self-Reflect   │  │
│                   ┌────────┐ ┌────────┐ ┌────────┐     │ (confidence    │  │
│                   │  RAG   │ │   KG   │ │  Tool  │     │  check, retry  │  │
│                   │  Node  │ │  Node  │ │  Node  │     │  if low)       │  │
│                   └────┬───┘ └────┬───┘ └────────┘     └───────┬────────┘  │
│                        │          │                             │           │
│                        ▼          ▼                    ┌───────▼────────┐  │
│                   ┌─────────────────────┐              │ Fact Extractor │  │
│                   │ Hybrid Fusion (RRF) │              │ + Memory Store │  │
│                   │ Dense + BM25 + Graph│              └────────────────┘  │
│                   └─────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼──────────────────────────────────────┐
│                           DATA & STORAGE LAYER                               │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │   ChromaDB   │  │    Neo4j     │  │  PostgreSQL  │  │   File System  │  │
│  │   + BM25     │  │  (+ NetworkX │  │  (+ SQLite   │  │                │  │
│  │              │  │   fallback)  │  │   fallback)  │  │  raw/ cleaned/ │  │
│  │  embeddings  │  │  entities    │  │  user_memory │  │  uploads/      │  │
│  │  chunks      │  │  relations   │  │  chat_history│  │                │  │
│  │  metadata    │  │  graph facts │  │  sessions    │  │                │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Offline data pipeline** (run once before serving):

```
URLs ──► scraper (requests + BS4) ──► cleaner (HTML → text) ──► chunker (500-word recursive split)
                                              │
                           ┌──────────────────┴──────────────────┐
                           ▼                                     ▼
            MiniLM embeddings ──► ChromaDB + BM25      LLM entity/relation extraction ──► Neo4j
            (dense + sparse index)                     (MERGE-deduplicated graph)
```

---
## 3. 🚦 End-to-End Query Flow

**Example Query:** `user_id="shami"` → *"Suggest a project I'd enjoy building."*

| Step  | Node                   | Action                                                                            |
| ----- | ---------------------- | --------------------------------------------------------------------------------- |
| **1** | `memory_node`          | Loads user memories and recent conversation history.                              |
| **2** | `rewrite_node`         | Rewrites the query using context for better understanding.                        |
| **3** | `router_node`          | Determines the best route (`direct`, `RAG`, `Knowledge Graph`, or `API`).         |
| **4** | `answer_node`          | Generates a personalized response using the selected route and available context. |
| **5** | `reflect_node`         | Evaluates response confidence and retries with deeper retrieval if needed.        |
| **6** | `fact_extraction_node` | Extracts and stores new long-term user facts when applicable.                     |

**Sample Response**

```json
{
  "answer": "Since you're interested in AI/ML and prefer Python, consider building...",
  "route": "direct",
  "sources": [],
  "confidence": 0.85,
  "provider": "groq",
  "latency_ms": 1020
}
```

Each response includes the selected **route**, **confidence score**, **sources**, and **latency**, making the system transparent and easy to debug.

---
## 4. 🧠 Dual Memory System

The assistant maintains **two complementary memory stores** to balance conversational context with long-term personalization.

|                 | 💬 `chat_history`           | 🧠 `user_memory`            |
| --------------- | --------------------------- | --------------------------- |
| **Purpose**     | Recent conversation context | Long-term user facts        |
| **Stores**      | Raw chat messages           | LLM-extracted durable facts |
| **Persistence** | Session history             | Across all sessions         |
| **Growth**      | Increases with conversation | Only new unique facts       |
| **Example**     | *"I prefer Python."*        | *"User prefers Python."*    |

### Why It Matters

Instead of relying only on the context window, important user preferences are stored as **persistent facts**. This keeps prompts compact while enabling personalization even across future sessions.

### Smart Memory Extraction

The system avoids unnecessary LLM calls by skipping extraction for messages that are:

* Too short (e.g., *"ok"*, *"thanks"*)
* Questions or commands
* Simple acknowledgements
* Messages without personal information

### Memory Versioning

When a user updates a preference (e.g., *"My favorite language is Rust"*), the previous fact is marked **inactive** and linked to the new one. This preserves a complete history without deleting data.

---
## 5. 🔍 Hybrid Retrieval Pipeline

Instead of relying only on vector similarity, the system combines **three complementary retrieval strategies** to improve relevance and recall.

### Retrieval Signals

| Signal     | Method                       | Best For                   |
| ---------- | ---------------------------- | -------------------------- |
| **Dense**  | ChromaDB + MiniLM embeddings | Semantic similarity        |
| **Sparse** | BM25                         | Exact keywords and phrases |
| **Graph**  | Neo4j traversal              | Entity relationships       |

### Reciprocal Rank Fusion (RRF)

Results from Dense and BM25 retrieval are merged using **Reciprocal Rank Fusion (RRF)**, which combines rankings instead of raw scores for more reliable retrieval.

```text
RRF(doc) = Σ 1 / (k + rank(doc))
```

The retrieval balance is configurable:

```env
HYBRID_ALPHA=0.7   # 70% Dense, 30% BM25
```

### Why RRF?

Unlike score-based fusion, **RRF is rank-based**, making it robust across retrieval methods with different scoring scales and consistently improving result quality.

---
## 6. 🔀 9-Node LangGraph Workflow

The assistant is powered by a **9-node LangGraph state machine** that dynamically routes each query to the most appropriate execution path.

```text
START
   │
memory_node
   │
rewrite_node
   │
router_node
   ├──► rag_node
   ├──► kg_node
   ├──► tool_node
   └──► direct
        │
    answer_node
        │
   reflect_node ──► Retry (max 2)
        │
fact_extraction_node
        │
       END
```

### 🔄 Self-Reflection Loop

Before returning a response, the workflow verifies that:

* The answer addresses the user's query.
* The confidence score meets the threshold.
* The response is grounded in retrieved context.

If any check fails, the graph performs **one additional retrieval cycle** (up to **2 retries**) before generating the final answer, improving reliability and response quality.

---
## 7. 🛠️ Dynamic Tool Calling

The agent can dynamically invoke external tools whenever additional information or computation is required.

| Tool                    | Purpose                                            |
| ----------------------- | -------------------------------------------------- |
| 🌐 **Web Search**       | Search the web for recent information and facts.   |
| 📖 **Wikipedia Lookup** | Retrieve concise summaries of well-known topics.   |
| 🌦️ **Weather**         | Fetch current weather conditions for any location. |
| 📈 **Market Data**      | Get live stock and cryptocurrency prices.          |
| 🧮 **Calculator**       | Perform mathematical calculations safely.          |
| 🐍 **Python Executor**  | Execute Python code in a sandboxed environment.    |

### 🔒 Safety Features

* **AST-based calculator** prevents unsafe code execution.
* **Sandboxed Python execution** with restricted imports and execution timeout.
* **Network tools** use request timeouts, retries, and error handling for reliable execution.

---
## 8. 📊 Evaluation Framework

The system is evaluated across **four complementary dimensions** to measure retrieval quality, response accuracy, and personalization.

### Evaluation Layers

* **🔍 Retrieval Metrics** — Hit Rate, MRR, Precision@K, Recall@K
* **📝 Response Quality** — Groundedness, Hallucination Rate, Answer Relevance, Context Utilization
* **🤖 LLM-as-Judge** — Faithfulness, Correctness, and Context Relevance (with embedding-based validation)
* **🧠 Memory Evaluation** — Memory Recall, Personalization Score, and Memory vs. Memoryless A/B testing

### Test Dataset

The evaluation includes **30+ curated test cases** covering:

* RAG & factual QA
* Knowledge Graph reasoning
* Tool/API queries
* Memory & personalization
* Multi-hop reasoning

---
## 9. 🧩 Core Components

| Module                    | Key Feature                                                          |
| ------------------------- | -------------------------------------------------------------------- |
| **LLM Engine**            | Multi-provider LLM orchestration with automatic failover.            |
| **Configuration**         | Centralized settings and validation.                                 |
| **Data Pipeline**         | Web scraping, document loading, cleaning, and chunking.              |
| **Embedding & Retrieval** | MiniLM embeddings with Hybrid Retrieval (ChromaDB + BM25 + RRF).     |
| **RAG Pipeline**          | End-to-end retrieval and grounded response generation.               |
| **Knowledge Graph**       | Entity extraction and graph-based reasoning with Neo4j.              |
| **Memory System**         | Persistent user memory with fact extraction and conflict handling.   |
| **Dynamic Tools**         | Safe tool execution for web search, weather, calculations, and more. |
| **Agent Workflow**        | 9-node LangGraph workflow with routing and self-reflection.          |
| **Evaluation**            | Multi-layer evaluation for retrieval, response quality, and memory.  |
| **FastAPI Backend**       | REST APIs with streaming support.                                    |
| **Streamlit Dashboard**   | Interactive UI for chat, memory, evaluation, and observability.      |

---

## 10. ⚙️ Tech Stack

| Layer                | Technology                      |
| -------------------- | ------------------------------- |
| **Language**         | Python 3.11+                    |
| **Backend**          | FastAPI                         |
| **Agent Framework**  | LangGraph                       |
| **LLMs**             | Groq, Google Gemini, Ollama     |
| **Embeddings**       | SentenceTransformers (MiniLM)   |
| **Vector Database**  | ChromaDB                        |
| **Sparse Retrieval** | BM25 (`rank-bm25`)              |
| **Knowledge Graph**  | Neo4j, NetworkX                 |
| **Memory Database**  | PostgreSQL, SQLite              |
| **Data Processing**  | BeautifulSoup, Requests, PyPDF2 |
| **Frontend**         | Streamlit                       |
| **Visualization**    | Plotly, PyVis, Streamlit-Agraph |
| **Testing**          | Pytest, HTTPX                   |
| **CLI Tools**        | Rich, tqdm                      |

---
## 11. 🚀 Setup & Run

### Prerequisites

* Python **3.11+**
* At least **one LLM provider** configured (Groq, Gemini, or Ollama)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Memory-Augmented-Chatbot.git
cd Memory-Augmented-Chatbot
```

### 2️⃣ Create & Activate a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

```bash
cp .env.example .env
# (Windows: copy .env.example .env)
```

Add your API key(s):

```env
GROQ_API_KEY= put_your_GROQ_key
GOOGLE_API_KEY= put_your_GEMINI_key
```

### 5️⃣ Ingest Knowledge Base

```bash
# From URLs
python scripts/ingest.py --file urls.txt

# Single URL
python scripts/ingest.py https://example.com

# PDF
python scripts/ingest.py --pdf document.pdf
```

### 6️⃣ Start the Application

```bash
# FastAPI
uvicorn app:app --reload

# Streamlit
streamlit run app_streamlit.py

# CLI
python scripts/chat_cli.py
```

Configure Neo4j and PostgreSQL in `.env` if using Docker.

---

## 12. 📡 API Reference

| Method | Endpoint                  | Purpose                                  |
| ------ | ------------------------- | ---------------------------------------- |
| `POST` | `/chat`                   | Chat with the assistant                  |
| `POST` | `/chat/stream`            | Streaming chat (SSE)                     |
| `GET`  | `/memory/{user_id}`       | Retrieve stored user memory              |
| `GET`  | `/chat/history/{user_id}` | View conversation history                |
| `POST` | `/data/ingest`            | Ingest documents into the knowledge base |
| `POST` | `/data/upload`            | Upload PDF/TXT documents                 |
| `GET`  | `/kg/entity/{name}`       | Query the Knowledge Graph                |
| `POST` | `/rag/retrieve`           | Test retrieval without generation        |
| `POST` | `/eval/run`               | Execute the evaluation suite             |
| `GET`  | `/health`                 | Check API and service health             |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
-H "Content-Type: application/json" \
-d '{"user_id":"Rudra","message":"What is a transformer in deep learning?"}'
```

### Example Response

```json
{
  "answer": "...",
  "route": "rag",
  "confidence": 0.92,
  "provider": "groq",
  "sources": ["Transformer", "Attention"],
  "latency_ms": 1092
}

---
## 13. 📁 Project Structure

```text
Memory-Augmented-Chatbot/
│
├── app.py                  # FastAPI backend
├── app_streamlit.py        # Streamlit UI
├── requirements.txt
├── .env.example
├── docker-compose.yml
│
├── src/
│   ├── llm/                # LLM orchestration & failover
│   ├── data/               # Data ingestion & preprocessing
│   ├── rag/                # Embeddings & hybrid retrieval
│   ├── graph/              # Knowledge Graph
│   ├── memory/             # Persistent memory system
│   ├── tools/              # Dynamic tools
│   ├── agent/              # LangGraph workflow
│   └── eval/               # Evaluation framework
│
├── scripts/                # Ingestion & CLI utilities
├── tests/                  # Unit & integration tests
├── notebooks/              # Experimentation notebooks
└── data/                   # Runtime data (DBs, graphs, documents)
```

---

## 14. ⚖️ Architecture Choices & Engineering Trade-offs

| Choice                           | Why It Was Chosen                                                       | Compromise                                               |
| -------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- |
| **LLM Failover Architecture**    | Uses multiple LLM providers for higher availability and reliability.    | Slightly more orchestration complexity.                  |
| **ChromaDB**                     | Lightweight, persistent, and easy to integrate with Python.             | Not as fast as FAISS for large-scale vector search.      |
| **Hybrid Retrieval**             | Combines semantic and keyword search for improved retrieval accuracy.   | Introduces a small retrieval overhead.                   |
| **Reciprocal Rank Fusion (RRF)** | Merges retrieval results consistently across different scoring methods. | Uses ranking instead of raw similarity scores.           |
| **Relational Memory Storage**    | PostgreSQL with SQLite fallback ensures reliable and persistent memory. | Less schema flexibility than NoSQL databases.            |
| **Selective Memory Extraction**  | Extracts user facts only when necessary, reducing LLM usage and cost.   | Rare edge-case facts may be skipped.                     |
| **Self-Reflection Workflow**     | Improves answer quality by retrying low-confidence responses.           | May increase latency for some queries.                   |
| **Neo4j with NetworkX Fallback** | Supports graph reasoning while remaining easy to run locally.           | Fallback graph operations are less efficient.            |
| **Local Embedding Model**        | MiniLM provides fast, free embeddings without external APIs.            | Slightly lower embedding quality than commercial models. |

---
## 15. 🔭 Limitations & Future Enhancements

### Current Limitations

* Supports multi-user memory but is primarily tested for single-user usage.
* API endpoints do not include authentication or authorization.
* Optimized for English-language interactions.
* Knowledge base performance has been evaluated on a moderate-sized dataset.
* Local LLM fallback requires Ollama to be installed and running.

### Future Enhancements

* Multi-hop Knowledge Graph reasoning for more complex queries.
* Conversation branching and versioning.
* JWT-based authentication and user management.
* Production monitoring with Prometheus and Grafana.
* Support for multilingual conversations and retrieval.

---

<div align="center">

**Built by Rudra Mehta for Celebal Technologies CEIP-2026**

</div>
