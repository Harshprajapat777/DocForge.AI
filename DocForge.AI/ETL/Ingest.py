#!/usr/bin/env python3
"""
ETL Pipeline - DocForge.AI
================================
Full PDF extraction via LlamaParse (cloud, AI-powered markdown).
  • Text   -> LlamaParse markdown -> chunked -> embedded -> ChromaDB
  • Tables -> LlamaParse markdown -> parsed  -> tables.json
"""

import os
import sys
import json
import io
import re
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent          # DocForge.AI/
ROOT_DIR   = BASE_DIR.parent                       # repo root
PDF_PATH   = BASE_DIR / "data" / "Report-Cyber-Ireland-2022.pdf"
CHROMA_DIR = BASE_DIR / "chromaDB"
TABLES_OUT = BASE_DIR / "data" / "tables.json"

load_dotenv(ROOT_DIR / ".env")

# ── Validate env ──────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
LLAMA_PARSE_KEY = os.getenv("llama_parser_key")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not set in .env")
    sys.exit(1)
if not LLAMA_PARSE_KEY:
    print("ERROR: llama_parser_key is not set in .env")
    sys.exit(1)

from llama_parse import LlamaParse
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
# STEP 1 - Full PDF extraction via LlamaParse
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf_with_llamaparse(pdf_path: Path) -> list:
    """
    Send PDF to LlamaParse cloud API.
    Returns list of raw LlamaParse Documents (one per page, markdown text).
    Result is cached in memory and reused by both text + table extraction.
    """
    print("      -> Sending PDF to LlamaParse API ...")
    parser = LlamaParse(
        api_key=LLAMA_PARSE_KEY,
        result_type="markdown",   # AI-parsed markdown: preserves structure, tables, headers
        verbose=False,
        language="en",
    )
    raw_docs = parser.load_data(str(pdf_path))
    print(f"      -> LlamaParse returned {len(raw_docs)} pages")
    return raw_docs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 - Build LlamaIndex Documents from parsed pages
# ─────────────────────────────────────────────────────────────────────────────

def build_text_documents(raw_docs: list) -> list:
    """
    Convert LlamaParse raw docs into LlamaIndex Documents for embedding.
    Metadata: page_number, source.
    """
    total     = len(raw_docs)
    documents = []
    for i, doc in enumerate(raw_docs, start=1):
        page_num = int(doc.metadata.get("page_label", i))
        text     = doc.text.strip() if doc.text else ""
        if len(text) < 30:
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "page_number": page_num,
                    "source":      f"Cyber Ireland 2022 Report - Page {page_num}",
                    "total_pages": total,
                },
            )
        )
    return documents


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 - Parse markdown tables from LlamaParse output -> structured JSON
# ─────────────────────────────────────────────────────────────────────────────

def _parse_markdown_table(block: str) -> dict | None:
    """
    Parse a single markdown table block into headers + rows dict.
    Returns None if the block is not a valid table.
    """
    lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
    # Need at least: header row + separator row + one data row
    table_lines = [l for l in lines if l.startswith("|")]
    if len(table_lines) < 3:
        return None

    def split_row(line: str) -> list:
        return [cell.strip() for cell in line.strip("|").split("|")]

    header_line = table_lines[0]
    sep_line    = table_lines[1]

    # Validate separator row (e.g. |---|---|)
    if not re.match(r"^\|[\s\-:|]+\|", sep_line):
        return None

    headers  = split_row(header_line)
    data_rows = [split_row(l) for l in table_lines[2:]]
    # Drop fully empty rows
    data_rows = [r for r in data_rows if any(c for c in r)]

    if not headers or not data_rows:
        return None

    return {"headers": headers, "rows": data_rows}


def extract_tables_from_llamaparse(raw_docs: list) -> list:
    """
    Extract and structure all markdown tables from LlamaParse output.
    Returns list of dicts: {page, table_index, title, headers, rows}.
    """
    all_tables = []

    for i, doc in enumerate(raw_docs, start=1):
        page_num = int(doc.metadata.get("page_label", i))
        text     = doc.text or ""

        # Split page text into blocks; look for markdown table blocks
        # A table block = consecutive lines starting with '|'
        blocks      = []
        current     = []
        prev_title  = ""

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                current.append(stripped)
            else:
                if current:
                    blocks.append((prev_title.strip(), "\n".join(current)))
                    current = []
                # Track last non-empty, non-table line as potential title
                if stripped and not stripped.startswith("#"):
                    prev_title = stripped
                elif stripped.startswith("#"):
                    prev_title = stripped.lstrip("#").strip()

        if current:
            blocks.append((prev_title.strip(), "\n".join(current)))

        for t_idx, (title, block) in enumerate(blocks):
            parsed = _parse_markdown_table(block)
            if not parsed:
                continue
            if not title:
                title = f"Table {t_idx + 1} (Page {page_num})"
            all_tables.append({
                "page":        page_num,
                "table_index": t_idx,
                "title":       title,
                "headers":     parsed["headers"],
                "rows":        parsed["rows"],
            })

    return all_tables


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 - Embed + Store in ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def build_chroma_store():
    """Create (or reset) ChromaDB collection and return LlamaIndex store + context."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION)
        print(f"      -> Dropped existing collection '{COLLECTION}'")
    except Exception:
        pass

    collection   = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx  = StorageContext.from_defaults(vector_store=vector_store)
    return vector_store, storage_ctx


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 - Validation (smoke-test retrieval)
# ─────────────────────────────────────────────────────────────────────────────

def validate(index, embed_model):
    """Run a quick retrieval test to confirm the index is working."""
    Settings.embed_model = embed_model
    retriever = index.as_retriever(similarity_top_k=3)
    nodes     = retriever.retrieve("total number of jobs cybersecurity Ireland")

    print(f"\n  Smoke-test — query: 'total number of jobs cybersecurity Ireland'")
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
    print(f"  DocForge.AI - ETL Pipeline")
    print(f"{sep}")
    print(f"  PDF    : {pdf_path.name}")
    print(f"  Parser : LlamaParse (full PDF — text + tables)")
    print(f"  Embed  : {EMBED_MODEL}")
    print(f"  Chunk  : {CHUNK_SIZE} tokens  |  Overlap: {CHUNK_OVERLAP} tokens")
    print(f"{sep}\n")

    # 1. Parse entire PDF via LlamaParse (single API call)
    print("[1/5] Parsing full PDF via LlamaParse ...")
    raw_docs = parse_pdf_with_llamaparse(pdf_path)

    # 2. Build text documents
    print("[2/5] Building text documents ...")
    documents = build_text_documents(raw_docs)
    print(f"      -> {len(documents)} pages with content")

    # 3. Extract tables from markdown
    print("[3/5] Extracting tables from LlamaParse markdown ...")
    tables = extract_tables_from_llamaparse(raw_docs)
    TABLES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLES_OUT, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"      -> {len(tables)} tables  ->  saved to {TABLES_OUT.name}")

    # 4. Embed + store in ChromaDB
    print("[4/5] Embedding and storing in ChromaDB ...")
    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    Settings.llm = None   # ETL does not need an LLM

    _, storage_ctx = build_chroma_store()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        show_progress=True,
    )

    # 5. Smoke-test
    print("[5/5] Validating ...")
    validate(index, embed_model)

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
