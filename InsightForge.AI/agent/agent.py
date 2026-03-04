# -*- coding: utf-8 -*-
"""
ReAct Agent - InsightForge.AI
==============================
Wires GPT-4o + 3 tools into a LlamaIndex ReAct agent.

Flow per query:
  Thought -> Action (pick tool) -> Observation (tool result) -> repeat -> Final Answer

Exports:
  run_query(query: str) -> dict
    {
      "query":       str,
      "answer":      str,
      "citations":   list[str],
      "tools_used":  list[str],
      "agent_steps": list[dict],
      "timestamp":   str
    }
"""

import os
import io
import sys
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# -- Paths --------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent.parent        # InsightForge.AI/
ROOT_DIR  = BASE_DIR.parent                     # InsightForgeAI/
LOGS_DIR  = BASE_DIR / "logs"
TRACES_FILE = LOGS_DIR / "traces.json"

load_dotenv(ROOT_DIR / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set in .env")
    sys.exit(1)

from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import sys
sys.path.insert(0, str(BASE_DIR))
from agent.tools import get_tools

# -- Config -------------------------------------------------------------------
LLM_MODEL   = "gpt-4o"
EMBED_MODEL = "text-embedding-3-small"

SYSTEM_PROMPT = """You are a precise research assistant for the Cyber Ireland 2022 Report.

STRICT RULES:
1. NEVER state a number, percentage, or statistic without first retrieving it using a tool.
2. ALWAYS use rag_search_tool to find text-based facts and page citations.
3. Use table_lookup_tool for tabular data FIRST, but if it returns no useful results after 2 attempts, ALWAYS fall back to rag_search_tool with different keywords.
4. ALWAYS use math_calculator_tool for any arithmetic — never compute numbers in your head.
5. Every factual claim in your final answer MUST include its page number as a citation.
6. For regional questions (South-West, Cork, Dublin, etc.): try BOTH table_lookup_tool AND rag_search_tool. Regional data may be in text paragraphs, not tables.
7. For CAGR questions: find baseline + target with tools first, then use math_calculator_tool.
8. If exact data is unavailable, clearly state what WAS found and derive the best answer from available data. Do NOT keep retrying the same failed search.
9. "Pure-Play" and "Dedicated" mean the same thing in this report — use both terms when searching.
10. "South-West" region includes Cork. Search for "Cork" as well as "South-West".

Response format:
- Lead with the direct answer
- Follow with exact citations: "Source: Page X — <exact quote>"
- If data is partially available, explain what was found and what was unavailable
"""


# =============================================================================
# Step logger — captures Thought / Action / Observation per ReAct step
# =============================================================================

class StepLogger:
    """Lightweight callback to capture ReAct agent steps for traces.json."""

    def __init__(self):
        self.steps: list[dict] = []
        self._current: dict = {}

    def on_event(self, event_type: str, payload: dict):
        """Called by LlamaIndex callback system."""
        if event_type == "llm":
            response = payload.get("response", {})
            if hasattr(response, "message"):
                content = str(response.message.content or "")
                # Parse Thought / Action / Action Input from ReAct output
                thought = self._extract(content, r"Thought:\s*(.+?)(?=\nAction:|$)")
                action  = self._extract(content, r"Action:\s*(.+?)(?=\nAction Input:|$)")
                inp     = self._extract(content, r"Action Input:\s*(.+?)(?=\n|$)")
                if thought or action:
                    self._current = {
                        "thought":      thought,
                        "action":       action,
                        "action_input": inp,
                        "observation":  None,
                    }

        elif event_type == "function_call":
            tool_name   = payload.get("function_call", "")
            tool_output = str(payload.get("tool_output", ""))
            if self._current:
                self._current["action"]      = tool_name
                self._current["observation"] = tool_output[:600]
                self.steps.append(dict(self._current))
                self._current = {}

    @staticmethod
    def _extract(text: str, pattern: str) -> str:
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""


# =============================================================================
# Citation extractor — pulls "Page X" references from answer text
# =============================================================================

def extract_citations(text: str) -> list[str]:
    """Find all 'Page N' mentions in the agent answer."""
    pages = re.findall(r"[Pp]age\s+(\d+)", text)
    seen, citations = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p)
            citations.append(f"Page {p}")
    return citations


def extract_tools_used(steps: list[dict]) -> list[str]:
    """Deduplicated list of tool names used across all steps."""
    seen, tools = set(), []
    for s in steps:
        t = s.get("action", "")
        if t and t not in seen:
            seen.add(t)
            tools.append(t)
    return tools


# =============================================================================
# Trace persistence
# =============================================================================

def save_trace(trace: dict) -> None:
    """Append this query's trace to logs/traces.json."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    existing = []
    if TRACES_FILE.exists() and TRACES_FILE.stat().st_size > 2:
        try:
            with open(TRACES_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append(trace)
    with open(TRACES_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# =============================================================================
# build_agent() — create ReAct agent (called once, reused per request)
# =============================================================================

def build_agent() -> ReActAgent:
    llm = OpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0,          # deterministic — factual Q&A
        max_tokens=2048,
    )

    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    Settings.llm         = llm
    Settings.embed_model = embed_model

    tools = get_tools()

    agent = ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,           # prints Thought/Action/Observation to stdout
        max_iterations=20,      # enough for multi-tool queries incl. fallback chains
        context=SYSTEM_PROMPT,
    )

    return agent


# =============================================================================
# run_query() — main public interface (called by FastAPI backend)
# =============================================================================

def run_query(query: str, agent: ReActAgent = None) -> dict:
    """
    Run a query through the ReAct agent.
    Returns structured dict with answer, citations, tools_used, agent_steps.
    Also appends the full trace to logs/traces.json.
    """
    if agent is None:
        agent = build_agent()

    print(f"\n{'='*62}")
    print(f"  Query: {query}")
    print(f"{'='*62}\n")

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        response = agent.chat(query)
        answer   = str(response)
    except Exception as e:
        answer = f"ERROR: Agent failed — {e}"

    # Pull step-level traces from agent's internal task list
    agent_steps = _extract_steps_from_agent(agent)

    citations  = extract_citations(answer)
    tools_used = extract_tools_used(agent_steps)

    trace = {
        "timestamp":   timestamp,
        "query":       query,
        "answer":      answer,
        "citations":   citations,
        "tools_used":  tools_used,
        "agent_steps": agent_steps,
    }

    save_trace(trace)

    print(f"\n{'='*62}")
    print(f"  Answer: {answer[:300]}...")
    print(f"  Citations : {citations}")
    print(f"  Tools used: {tools_used}")
    print(f"{'='*62}\n")

    return trace


def _extract_steps_from_agent(agent: ReActAgent) -> list[dict]:
    """
    Extract Thought/Action/Observation steps from the agent's
    completed task sources. Works with LlamaIndex 0.10.x ReActAgent.
    """
    steps = []
    try:
        # ReActAgent stores completed tasks in _task_dict (internal)
        for task_id, task in agent._task_dict.items():
            for step_output in task.completed_steps:
                output = step_output.output
                if hasattr(output, "sources"):
                    for src in output.sources:
                        steps.append({
                            "thought":      getattr(src, "thought", ""),
                            "action":       getattr(src, "tool_name", ""),
                            "action_input": str(getattr(src, "tool_input", "")),
                            "observation":  str(getattr(src, "content", ""))[:600],
                        })
    except Exception:
        # Fallback: return empty — answer is still correct, just no step detail
        pass
    return steps


# =============================================================================
# CLI test — run the 3 assignment queries directly
# =============================================================================

if __name__ == "__main__":
    agent = build_agent()

    TEST_QUERIES = [
        "What is the total number of jobs reported, and where exactly is this stated?",
        "Compare the concentration of Pure-Play cybersecurity firms in the South-West against the National Average.",
        "Based on our 2022 baseline and the stated 2030 job target, what is the required compound annual growth rate (CAGR) to hit that goal?",
    ]

    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n{'#'*62}")
        print(f"  TEST {i}")
        print(f"{'#'*62}")
        result = run_query(q, agent)
        print(f"\nFINAL ANSWER:\n{result['answer']}\n")
        print(f"Citations : {result['citations']}")
        print(f"Tools     : {result['tools_used']}")
