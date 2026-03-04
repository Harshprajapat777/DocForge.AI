## 🏗️ Complete Pipeline

### **Phase 1 — ETL**

```

cyber_ireland_2022.pdf
↓
pdfplumber
├── Text extract → page wise
│ ↓
│ chunk_size=300, overlap=50
│ ↓
│ + metadata: {page_number, source}
│ ↓
│ OpenAI Embeddings
│ ↓
│ ChromaDB (persist)
│
└── Tables extract → per page
↓
Clean + Structure
↓
tables.json save

```

---

### **Phase 2 — Agent Flow**

```

POST /chat {"query": "..."}
↓
LlamaIndex Agent
↓
Decides which tool:
├── Tool 1: RAG Search Tool
│ → ChromaDB se relevant chunks
│ → page_number + citation return
│
├── Tool 2: Table Lookup Tool
│ → tables.json se exact data
│ → South-West, National etc
│
└── Tool 3: Math Calculator Tool
→ CAGR formula
→ Pure Python execute karo
↓
GPT-4o → Final Answer synthesize
↓
Response JSON return

{
"answer": "Total jobs: 6,930...",
"citation": "Page 12 — exact line...",
"agent_steps": [
"Tool used: rag_search_tool",
"Tool used: table_lookup_tool",
"Tool used: math_calculator_tool"
]
}

## 🔄 Data Flow — Ek Line Mein Per Phase

```

[PDF] → ETL → [ChromaDB + tables.json]
↓
[Query] → FastAPI → Agent → Tools → GPT-4o → [Answer + Citation + Logs]
↓
[static/index.html] ← Optional UI

```

```
