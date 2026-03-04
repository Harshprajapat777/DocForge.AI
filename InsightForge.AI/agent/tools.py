# -*- coding: utf-8 -*-
"""
Agent Tools - InsightForge.AI
==============================
Three tools registered to the ReAct agent:

  Tool 1 | rag_search_tool      -> semantic search over ChromaDB
                                   returns: page_number + exact text (citation)

  Tool 2 | table_lookup_tool    -> keyword search over tables.json
                                   returns: exact table rows (no hallucination risk)

  Tool 3 | math_calculator_tool -> pure Python arithmetic
                                   returns: CAGR / percent diff / basic math
"""

import os
import io
import sys
import json
import math
from pathlib import Path

from dotenv import load_dotenv

# Force UTF-8 on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # InsightForge.AI/
ROOT_DIR    = BASE_DIR.parent                       # InsightForgeAI/
CHROMA_DIR  = BASE_DIR / "chromaDB"
TABLES_PATH = BASE_DIR / "data" / "tables.json"

load_dotenv(ROOT_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
COLLECTION  = "cyber_ireland_2022"
TOP_K       = 8          # No reranker — single PDF, top_k=8 is sufficient


# ==============================================================================
# TOOL 1 — RAG Search (semantic retrieval with page citations)
# ==============================================================================

def rag_search_tool(query: str) -> str:
    """
    Search the Cyber Ireland 2022 report for facts, statistics, and statements.
    Returns the top matching text passages with exact page numbers and source citations.
    Use this tool when you need:
      - Specific numbers or figures mentioned in the report
      - Exact quotes or statements to verify
      - Page-level citations for any factual claim
      - Employment figures, company counts, growth projections
    Input: a natural language search query string.
    """
    from llama_index.core import VectorStoreIndex, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb

    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    Settings.embed_model = embed_model
    Settings.llm = None

    client       = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection   = client.get_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index        = VectorStoreIndex.from_vector_store(
                        vector_store, embed_model=embed_model
                   )

    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes     = retriever.retrieve(query)

    if not nodes:
        return "No relevant content found for this query."

    results = []
    for i, node in enumerate(nodes, 1):
        page   = node.metadata.get("page_number", "Unknown")
        source = node.metadata.get("source", "Cyber Ireland 2022")
        score  = round(node.score, 4) if node.score else "n/a"
        results.append(
            f"[Result {i}] Page {page} (relevance: {score})\n"
            f"Source : {source}\n"
            f"Text   : {node.text.strip()}\n"
        )

    return "\n---\n".join(results)


# ==============================================================================
# TOOL 2 — Table Lookup (exact keyword search over tables.json)
# ==============================================================================

def table_lookup_tool(keywords: str) -> str:
    """
    Search structured tables extracted from the Cyber Ireland 2022 report.
    Returns exact table headers and rows as they appear in the document.
    Use this tool when you need:
      - Regional breakdowns (South-West, Dublin, Munster, etc.)
      - Pure-Play vs non-Pure-Play firm percentages
      - National Average comparisons
      - Any numerical data organised in a table format
    Input: one or more keywords separated by spaces or commas.
    Example inputs: "Pure-Play South-West", "National Average", "regional"
    """
    if not TABLES_PATH.exists():
        return "ERROR: tables.json not found. Run ETL/Ingest.py first."

    with open(TABLES_PATH, "r", encoding="utf-8") as f:
        tables = json.load(f)

    search_terms = [
        k.strip().lower()
        for k in keywords.replace(",", " ").split()
        if k.strip()
    ]

    if not search_terms:
        return "Please provide at least one keyword to search."

    matches = []
    for table in tables:
        title_text   = table.get("title", "").lower()
        headers_text = " ".join(str(h) for h in table.get("headers", [])).lower()
        rows_text    = " ".join(
            " ".join(str(c) for c in row)
            for row in table.get("rows", [])
        ).lower()

        combined = f"{title_text} {headers_text} {rows_text}"

        if any(term in combined for term in search_terms):
            matches.append(table)

    if not matches:
        return (
            f"No tables found matching: '{keywords}'\n"
            f"Tip: Try broader keywords or use rag_search_tool for text-based lookup."
        )

    output = []
    for t in matches:
        header_line = " | ".join(str(h) for h in t["headers"])
        row_lines   = [
            "  " + " | ".join(str(c) for c in row)
            for row in t["rows"]
        ]
        block = (
            f"Page {t['page']} | {t['title']}\n"
            f"Headers : {header_line}\n"
            + "\n".join(row_lines)
        )
        output.append(block)

    return f"Found {len(matches)} matching table(s):\n\n" + "\n\n---\n\n".join(output)


# ==============================================================================
# TOOL 3 — Math Calculator (pure Python — never let the LLM do arithmetic)
# ==============================================================================

def math_calculator_tool(input_json: str) -> str:
    """
    Execute precise mathematical calculations. ALWAYS use this tool for any arithmetic.
    Never compute numbers mentally — LLMs make arithmetic errors.

    Supported operations:

    1. CAGR (Compound Annual Growth Rate):
       Input: {"operation": "cagr", "start_value": 6930, "end_value": 17000, "years": 8}
       Formula: ((end_value / start_value) ^ (1 / years) - 1) * 100

    2. Percentage difference between two values:
       Input: {"operation": "percent_diff", "value_a": 14, "value_b": 22}
       Formula: ((b - a) / a) * 100

    3. Basic arithmetic expression:
       Input: {"operation": "basic", "expression": "6930 * 1.05 ** 8"}

    Input must be a valid JSON string.
    """
    # ── Parse input ────────────────────────────────────────────────────────────
    try:
        params = json.loads(input_json)
    except json.JSONDecodeError as e:
        return (
            f"ERROR: Invalid JSON input — {e}\n"
            f"Example: {{\"operation\": \"cagr\", \"start_value\": 6930, "
            f"\"end_value\": 17000, \"years\": 8}}"
        )

    operation = params.get("operation", "").lower().strip()

    # ── CAGR ───────────────────────────────────────────────────────────────────
    if operation == "cagr":
        try:
            start = float(params["start_value"])
            end   = float(params["end_value"])
            years = float(params["years"])
        except KeyError as e:
            return f"ERROR: Missing field {e}. Required: start_value, end_value, years"
        except (ValueError, TypeError) as e:
            return f"ERROR: All values must be numbers. {e}"

        if start <= 0:
            return "ERROR: start_value must be > 0"
        if years <= 0:
            return "ERROR: years must be > 0"

        cagr = (math.pow(end / start, 1.0 / years) - 1.0) * 100.0

        return (
            f"CAGR Result:\n"
            f"  Formula     : ((end / start) ^ (1 / years) - 1) * 100\n"
            f"  start_value : {start:,.0f}\n"
            f"  end_value   : {end:,.0f}\n"
            f"  years       : {years:.0f}\n"
            f"  CAGR        : {cagr:.4f}% per year\n"
            f"  Rounded     : {cagr:.2f}%\n"
            f"  Meaning     : Growing from {start:,.0f} to {end:,.0f} over "
            f"{years:.0f} years requires {cagr:.2f}% annual compound growth."
        )

    # ── Percentage Difference ──────────────────────────────────────────────────
    elif operation == "percent_diff":
        try:
            a = float(params["value_a"])
            b = float(params["value_b"])
        except KeyError as e:
            return f"ERROR: Missing field {e}. Required: value_a, value_b"
        except (ValueError, TypeError) as e:
            return f"ERROR: All values must be numbers. {e}"

        if a == 0:
            return "ERROR: value_a cannot be 0 (division by zero)"

        diff = ((b - a) / a) * 100

        return (
            f"Percentage Difference Result:\n"
            f"  Formula : ((b - a) / a) * 100\n"
            f"  value_a : {a}\n"
            f"  value_b : {b}\n"
            f"  Result  : {diff:+.2f}%\n"
            f"  Meaning : b is {diff:+.2f}% {'higher' if diff >= 0 else 'lower'} than a."
        )

    # ── Basic Expression ───────────────────────────────────────────────────────
    elif operation == "basic":
        expr = params.get("expression", "").strip()
        if not expr:
            return "ERROR: 'expression' field is empty."

        try:
            safe_globals = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            result = eval(expr, {"__builtins__": {}}, safe_globals)
            return (
                f"Basic Calculation Result:\n"
                f"  Expression : {expr}\n"
                f"  Result     : {result}"
            )
        except Exception as e:
            return f"ERROR evaluating '{expr}': {e}"

    # ── Unknown operation ──────────────────────────────────────────────────────
    else:
        return (
            f"ERROR: Unknown operation '{operation}'.\n"
            f"Supported: 'cagr', 'percent_diff', 'basic'"
        )


# ==============================================================================
# get_tools() — called by agent.py to register tools with ReAct agent
# ==============================================================================

def get_tools():
    """Return all three tools as LlamaIndex FunctionTool objects."""
    from llama_index.core.tools import FunctionTool

    return [
        FunctionTool.from_defaults(
            fn=rag_search_tool,
            name="rag_search_tool",
            description=(
                "Search the Cyber Ireland 2022 report for facts, statistics, and text passages. "
                "Returns exact text with page numbers for citation. "
                "Use for: job numbers, employment figures, company counts, "
                "policy statements, growth projections, any text-based facts."
            ),
        ),
        FunctionTool.from_defaults(
            fn=table_lookup_tool,
            name="table_lookup_tool",
            description=(
                "Search structured tables extracted from the report using keywords. "
                "Returns exact headers and row data. "
                "Use for: Pure-Play percentages, regional stats (South-West, Dublin), "
                "National Average comparisons, any data in table format."
            ),
        ),
        FunctionTool.from_defaults(
            fn=math_calculator_tool,
            name="math_calculator_tool",
            description=(
                "Execute precise mathematical calculations. "
                "ALWAYS use this tool for any arithmetic — never compute numbers mentally. "
                "Supports: CAGR, percentage difference, basic expressions. "
                "Input must be a valid JSON string with 'operation' and required parameters."
            ),
        ),
    ]
