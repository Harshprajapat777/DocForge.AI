# -*- coding: utf-8 -*-
"""
FastAPI Backend - InsightForge.AI
===================================
Endpoints:
  POST /chat    -> run agent query, return answer + citations + steps
  GET  /health  -> liveness check
  GET  /        -> serve chat UI (static/templates/home.html)

Agent is built once on startup (lifespan), reset between requests.
Sync agent.chat() runs in a thread pool to avoid blocking the event loop.
"""

import os
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# -- Paths & imports ----------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent           # InsightForge.AI/
ROOT_DIR   = BASE_DIR.parent                        # InsightForgeAI/
STATIC_DIR = BASE_DIR / "static"

load_dotenv(ROOT_DIR / ".env")

# Make agent module importable
sys.path.insert(0, str(BASE_DIR))
from agent.agent import build_agent, run_query

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    query: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "What is the total number of jobs reported?"},
                {"query": "Compare Pure-Play firms in South-West vs National Average."},
                {"query": "What CAGR is needed to hit the 2030 job target?"},
            ]
        }
    }


class QueryResponse(BaseModel):
    query:       str
    answer:      str
    citations:   list[str]
    tools_used:  list[str]
    agent_steps: list[dict]
    timestamp:   str


# =============================================================================
# App lifespan — build agent once on startup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[startup] InsightForge.AI is ready. Agent built per-request.")
    print("[startup] nest_asyncio applied — event loop conflicts resolved.\n")
    yield
    print("\n[shutdown] Cleaning up...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="InsightForge.AI",
    description="Autonomous RAG agent for the Cyber Ireland 2022 Report",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local dev / demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# Routes
# =============================================================================

@app.get("/health", tags=["System"])
async def health():
    """Liveness check — confirms server + agent are running."""
    return {
        "status":     "ok",
        "llm":        "gpt-4o",
        "embeddings": "text-embedding-3-small",
        "vector_db":  "chromadb (local)",
        "parser":     "LlamaParse (markdown)",
    }


@app.post("/chat", response_model=QueryResponse, tags=["Agent"])
async def chat(request: QueryRequest):
    """
    Run a natural language query through the ReAct agent.

    The agent autonomously decides which tools to use:
    - rag_search_tool       for text-based facts + page citations
    - table_lookup_tool     for regional/tabular data
    - math_calculator_tool  for CAGR and arithmetic

    Returns the final answer, page citations, tool trace, and agent steps.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Build a fresh agent per request — avoids stale memory / event loop conflicts.
    # nest_asyncio (applied in run.py) allows LlamaIndex's internal asyncio.run()
    # calls to work correctly inside FastAPI's running event loop.
    def _run(q: str) -> dict:
        agent = build_agent()
        return run_query(q, agent)

    try:
        result = await asyncio.to_thread(_run, query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return QueryResponse(**result)


@app.get("/", tags=["UI"], include_in_schema=False)
async def serve_ui():
    """Serve the chat UI."""
    html_path = STATIC_DIR / "templates" / "home.html"
    if not html_path.exists():
        return JSONResponse(
            {"message": "InsightForge.AI is running. POST /chat to query the agent."},
            status_code=200,
        )
    return FileResponse(str(html_path))


# Mount static assets (CSS, JS) and logo
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

LOGO_DIR = BASE_DIR / "logo"
if LOGO_DIR.exists():
    app.mount("/logo", StaticFiles(directory=str(LOGO_DIR)), name="logo")
