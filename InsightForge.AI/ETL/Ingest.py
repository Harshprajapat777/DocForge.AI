#!/usr/bin/env python3
"""
ETL Pipeline - InsightForge.AI
================================
Extracts text + tables from Cyber Ireland 2022 PDF.
  • Text   -> chunked (400t / 50t overlap) -> embedded -> ChromaDB
  • Tables -> cleaned + structured         -> tables.json
"""

import os
import sys
import json
import io
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # InsightForge.AI/
ROOT_DIR   = BASE_DIR.parent                       # InsightForgeAI/
PDF_PATH   = BASE_DIR / "data" / "Report-Cyber-Ireland-2022.pdf"
CHROMA_DIR = BASE_DIR / "chromaDB"
TABLES_OUT = BASE_DIR / "data" / "tables.json"

load_dotenv(ROOT_DIR / ".env")

# ── Validate env ──────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not set in .env")
    sys.exit(1)

import pdfplumber
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
EMBED_MODEL   = "text-embedding-3-small"
COLLECTION    = "cyber_ireland_2022"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 - Text Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_documents(pdf_path: Path) -> list:
    """
    Extract page-level text as LlamaIndex Documents.
    Each Document carries metadata: page_number, source.
    """
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or len(text.strip()) < 30:
                continue
            documents.append(
                Document(
                    text=text.strip(),
                    metadata={
                        "page_number": page_num,
                        "source": f"Cyber Ireland 2022 Report - Page {page_num}",
                        "total_pages": total,
                    },
                )
            )
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 - Table Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _clean_row(row: list) -> list:
    """Strip whitespace and replace None cells with empty string."""
    return [str(cell).strip() if cell is not None else "" for cell in row]


def _get_title_above_table(page, table_bbox: tuple) -> str:
    """
    Find the text line immediately above a table using its bounding box.
    table_bbox = (x0, top, x1, bottom)
    """
    table_top = table_bbox[1]
    words = page.extract_words()
    # Words whose bottom edge sits above the table top
    above = [w for w in words if w["bottom"] <= table_top]
    if not above:
        return ""
    # Sort by vertical position, take the last (closest) line
    above.sort(key=lambda w: w["top"])
    closest_y = above[-1]["top"]
    line_words = [w for w in above if abs(w["top"] - closest_y) < 6]
    line_words.sort(key=lambda w: w["x0"])
    return " ".join(w["text"] for w in line_words)


def extract_tables(pdf_path: Path) -> list:
    """
    Extract all tables from PDF using pdfplumber's find_tables() API.
    Returns a list of structured dicts with page, title, headers, rows.
    """
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            found = page.find_tables()
            if not found:
                continue

            for t_idx, table_obj in enumerate(found):
                raw = table_obj.extract()
                if not raw or len(raw) < 2:
                    continue

                # Clean rows, drop fully-empty rows
                cleaned = [_clean_row(row) for row in raw]
                cleaned = [r for r in cleaned if any(c for c in r)]
                if len(cleaned) < 2:
                    continue

                headers = cleaned[0]
                rows    = cleaned[1:]

                # Attempt to find the table's descriptive title
                title = _get_title_above_table(page, table_obj.bbox)
                if not title:
                    title = f"Table {t_idx + 1} (Page {page_num})"

                all_tables.append({
                    "page":        page_num,
                    "table_index": t_idx,
                    "title":       title,
                    "headers":     headers,
                    "rows":        rows,
                })

    return all_tables


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 - Embed + Store in ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def build_chroma_store():
    """Create (or reset) ChromaDB collection and return LlamaIndex store + context."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Clean slate on every ingest run
    try:
        client.delete_collection(COLLECTION)
        print(f"      -> Dropped existing collection '{COLLECTION}'")
    except Exception:
        pass

    collection    = client.get_or_create_collection(COLLECTION)
    vector_store  = ChromaVectorStore(chroma_collection=collection)
    storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_ctx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 - Validation (quick retrieval smoke-test)
# ─────────────────────────────────────────────────────────────────────────────

def validate(index, embed_model):
    """Run a quick retrieval test to confirm the index is working."""
    Settings.embed_model = embed_model
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("total number of jobs cybersecurity Ireland")

    print(f"\n  Smoke-test - query: 'total number of jobs cybersecurity Ireland'")
    for i, node in enumerate(nodes, 1):
        pg      = node.metadata.get("page_number", "?")
        preview = node.text[:140].replace("\n", " ")
        score   = round(node.score, 4) if node.score else "n/a"
        print(f"    [{i}] Page {pg} (score={score}): {preview}...")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def ingest(pdf_path: Path) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  InsightForge.AI - ETL Pipeline")
    print(f"{sep}")
    print(f"  PDF    : {pdf_path.name}")
    print(f"  Embed  : {EMBED_MODEL}")
    print(f"  Chunk  : {CHUNK_SIZE} tokens  |  Overlap: {CHUNK_OVERLAP} tokens")
    print(f"  top_k  : 8 (no reranker - single PDF corpus)")
    print(f"{sep}\n")

    # 1. Text
    print("[1/4] Extracting text ...")
    documents = extract_text_documents(pdf_path)
    print(f"      -> {len(documents)} pages with content")

    # 2. Tables
    print("[2/4] Extracting tables ...")
    tables = extract_tables(pdf_path)
    TABLES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLES_OUT, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"      -> {len(tables)} tables  ->  saved to {TABLES_OUT.name}")

    # 3. Embed + Store
    print("[3/4] Embedding and storing in ChromaDB ...")
    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    # Suppress LLM warnings - ETL doesn't need one
    Settings.llm = None

    _, storage_ctx = build_chroma_store()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=True,
    )

    # 4. Validate
    print("[4/4] Validating ...")
    validate(index, embed_model)

    # Summary
    print(f"\n{sep}")
    print(f"  ETL Complete")
    print(f"{sep}")
    print(f"  Pages ingested : {len(documents)}")
    print(f"  Tables saved   : {len(tables)}")
    print(f"  ChromaDB       : {CHROMA_DIR}")
    print(f"  Tables JSON    : {TABLES_OUT}")
    print(f"{sep}\n")


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)
    ingest(PDF_PATH)
