# -*- coding: utf-8 -*-
"""
run.py - InsightForge.AI server launcher
Run from project root: python run.py

The folder name 'InsightForge.AI' contains a dot which breaks Python's
standard module import (uvicorn "InsightForge.AI.backend.main:app" fails).
This script adds the app directory to sys.path and launches uvicorn directly.
"""

import sys
import nest_asyncio
from pathlib import Path

# Must apply BEFORE uvicorn/FastAPI start.
# LlamaIndex embedding calls use asyncio.run() internally which conflicts
# with FastAPI's running event loop when invoked via asyncio.to_thread.
# nest_asyncio allows nested event loops — fixes "Reached max iterations" on UI.
nest_asyncio.apply()

# Add InsightForge.AI/ to path so 'backend.main' resolves correctly
APP_DIR = Path(__file__).parent / "InsightForge.AI"
sys.path.insert(0, str(APP_DIR))

import uvicorn
from backend.main import app  # noqa: E402 — must come after sys.path insert

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,       # reload=True breaks with dynamic sys.path
        log_level="info",
    )
