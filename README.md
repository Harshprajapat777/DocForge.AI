# InsightForge.AI

Autonomous RAG agent for the **Cyber Ireland 2022 Report**. Ask natural language questions and get cited, tool-grounded answers — no hallucinated numbers.

---

## What It Does

- Parses the Cyber Ireland 2022 PDF (text + tables) into a ChromaDB vector store
- Runs a **LlamaIndex ReAct agent** (GPT-4o) that autonomously selects tools per query
- Returns answers with **exact page citations**, tools used, and agent reasoning steps
- Full chat UI at `http://localhost:8001` — reasoning accordion, citations, and tool badges included

---

## Prerequisites

You need **two API keys** before anything works:

### 1. OpenAI API Key
Used for GPT-4o (reasoning) and `text-embedding-3-small` (embeddings).

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click **Create new secret key** → copy the key (`sk-...`)

### 2. LlamaParse API Key
Used to parse the PDF into clean markdown with inline tables.

1. Go to [cloud.llamaindex.ai](https://cloud.llamaindex.ai)
2. Sign up → **API Keys** → create and copy the key (`llx-...`)

---

## Setup

### Step 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd InsightForgeAI

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### Step 2 — Install dependencies

```bash
pip install -r Requirements.txt
```

### Step 3 — Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-openai-key-here
LLAMA_CLOUD_API_KEY=llx-your-llamaparse-key-here
```

### Step 4 — Run ETL (one-time)

Parses the PDF with LlamaParse, embeds chunks into ChromaDB, and extracts tables to `tables.json`.

```bash
# Run from the project root (InsightForgeAI/)
python InsightForge.AI/ETL/Ingest.py
```

Expected output:
```
[1/4] Parsing PDF with LlamaParse ...  -> 39 pages parsed
[2/4] Extracting tables ...            -> 22 tables -> data/tables.json
[3/4] Chunking, embedding, storing in ChromaDB ...
[4/4] Validating ...

ETL Complete
  Pages parsed : 39
  Tables saved : 22
  ChromaDB     : InsightForge.AI/chromaDB/
  Tables JSON  : InsightForge.AI/data/tables.json
```

### Step 5 — Start the server

```bash
# From the project root (InsightForgeAI/)
python run.py
```

Open **`http://localhost:8001`** — the three assignment queries are pre-loaded in the sidebar.

Each query appends a full trace to `logs/traces.json` (answer, citations, tools used, tool observations). The full Thought → Action → Observation chain is also printed live in the server terminal.

---

## Test Queries

The three assignment scenarios, available as one-click sidebar chips in the UI:

| # | Query | Tools Invoked |
|---|-------|--------------|
| 1 | What is the total number of jobs reported, and where exactly is this stated? | `rag_search_tool` |
| 2 | Compare the concentration of Pure-Play cybersecurity firms in the South-West against the National Average | `table_lookup_tool` → `rag_search_tool` |
| 3 | Based on the 2022 baseline and the stated 2030 job target, what is the required CAGR to hit that goal? | `rag_search_tool` → `math_calculator_tool` |

---

## Project Structure

```
InsightForgeAI/
├── InsightForge.AI/
│   ├── ETL/
│   │   └── Ingest.py              # PDF -> ChromaDB + tables.json (LlamaParse)
│   ├── agent/
│   │   ├── tools.py               # 3 tools: rag_search, table_lookup, math_calculator
│   │   └── agent.py               # LlamaIndex ReActAgent + trace logger
│   ├── backend/
│   │   └── main.py                # FastAPI: /chat, /health, serves UI
│   ├── data/
│   │   ├── Report-Cyber-Ireland-2022.pdf
│   │   └── tables.json            # Structured table data (ETL output)
│   ├── chromaDB/                  # Vector store (ETL output)
│   ├── logs/
│   │   └── traces.json            # Per-query agent traces (auto-generated)
│   ├── logo/
│   │   └── InsightAI.jpg
│   └── static/
│       ├── templates/home.html    # Chat UI
│       ├── styles/home.css
│       └── Main.js
├── run.py                         # Server launcher (handles dot in folder name)
├── .env                           # API keys — gitignored
├── .env.example                   # Key template
├── Requirements.txt
└── implementation.md              # Architecture planning notes
```

---

## Architecture

| Layer | Choice | Why |
|-------|--------|-----|
| PDF Parser | LlamaParse | Returns clean markdown per page; tables preserved inline |
| Embeddings | `text-embedding-3-small` | 8191-token context window; $0.00002/1K — cheaper than ada-002 and avoids MiniLM's 256-token truncation |
| Vector DB | ChromaDB (local) | Persistent, no infrastructure overhead, metadata filtering for page citations |
| Agent | LlamaIndex ReAct | Native tool orchestration, step-level traces, first-class document metadata |
| LLM | GPT-4o | Required for reliable multi-step reasoning across ambiguous queries |
| Table storage | `tables.json` (separate from vector DB) | Deterministic keyword lookup — avoids semantic drift on numeric data |
| Math | Pure Python tool | LLMs are unreliable at CAGR arithmetic; Python guarantees correctness |

**Why LlamaIndex over LangChain:** LlamaIndex has first-class `page_number` metadata on every document node, which is essential for generating verifiable citations. Its ReAct agent natively exposes tool call traces per step.

**Why no reranker:** Single 39-page PDF (~91 chunks). `top_k=8` retrieval + GPT-4o self-selection is sufficient. A cross-encoder reranker (`FlashrankRerank`) would add latency with no meaningful gain at this scale — worth adding at 10+ documents.

---

## Execution Traces

Every query is logged to `logs/traces.json`:

```json
{
  "timestamp": "2026-03-04T12:00:00Z",
  "query": "What is the total number of jobs reported?",
  "answer": "7,351 employees...",
  "citations": ["Page 23"],
  "tools_used": ["rag_search_tool"],
  "agent_steps": [
    {
      "action": "rag_search_tool",
      "action_input": "total number of jobs cybersecurity Ireland",
      "observation": "[Result 1] Page 23 (relevance: 0.91)\nText: ...7,351 employees..."
    }
  ]
}
```

The full Thought / Action / Observation chain for each step is also printed to the server terminal in real time.

---

## API Reference

### `POST /chat`

```json
// Request
{ "query": "What is the total number of jobs reported?" }

// Response
{
  "query": "...",
  "answer": "...",
  "citations": ["Page 23"],
  "tools_used": ["rag_search_tool"],
  "agent_steps": [{ "action": "rag_search_tool", "action_input": "...", "observation": "..." }],
  "timestamp": "2026-03-04T12:00:00Z"
}
```

### `GET /health`

```json
{ "status": "ok", "llm": "gpt-4o", "embeddings": "text-embedding-3-small", "vector_db": "chromadb (local)" }
```

Interactive docs: `http://localhost:8001/docs`

---

## Limitations

- **Single document:** ChromaDB collection is hardcoded to Cyber Ireland 2022; would need a collection-per-document model to scale
- **Chart data:** Some report figures are embedded as images — LlamaParse cannot extract them; the agent derives answers from available text in those cases
- **OpenAI dependency:** No offline fallback; both LLM and embeddings require a valid API key and internet access
- **No reranker:** Intentional for this scale; `FlashrankRerank` would be added at 10+ documents
- **Sync ETL:** `Ingest.py` is a blocking single-process script; a multi-PDF corpus would need async chunking and a document queue
- **Production scaling:** ChromaDB local → Pinecone or Weaviate; single-agent → LangGraph or CrewAI for parallelised tool calls
