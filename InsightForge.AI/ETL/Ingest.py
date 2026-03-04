# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ETL Pipeline - InsightForge.AI  [LlamaParse Edition]
======================================================
Replaces pdfplumber text extraction with LlamaParse:
  - Returns markdown per page (tables as | col | col | inline)
  - Correct reading order, proper table titles
  - Chunked + embedded with text-embedding-3-small -> ChromaDB
  - Markdown tables parsed -> tables.json (deterministic lookup)
"""

import os
import io
import sys
import re
import json
from pathlib import Path

# Force UTF-8 on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# -- Paths --------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent       # InsightForge.AI/
ROOT_DIR   = BASE_DIR.parent                    # InsightForgeAI/
PDF_PATH   = BASE_DIR / "data" / "Report-Cyber-Ireland-2022.pdf"
CHROMA_DIR = BASE_DIR / "chromaDB"
TABLES_OUT = BASE_DIR / "data" / "tables.json"

load_dotenv(ROOT_DIR / ".env")

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set in .env")
    sys.exit(1)
if not LLAMA_CLOUD_API_KEY:
    print("ERROR: LLAMA_CLOUD_API_KEY not set in .env")
    sys.exit(1)

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# -- Config -------------------------------------------------------------------
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
EMBED_MODEL   = "text-embedding-3-small"
COLLECTION    = "cyber_ireland_2022"


# =============================================================================
# STEP 1 - LlamaParse: PDF -> Markdown Documents (one per page)
# =============================================================================

def parse_pdf_with_llamaparse(pdf_path: Path) -> list:
    """
    Send PDF to LlamaParse API.
    Returns list of LlamaIndex Documents, one per page, in markdown format.
    Tables come back as proper | col | col | markdown tables with titles.
    """
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",     # preserves table structure inline
        verbose=True,
        language="en",
    )

    raw_docs = parser.load_data(str(pdf_path))

    # Normalise metadata: add page_number (int) and source string
    for i, doc in enumerate(raw_docs):
        # LlamaParse sets page_label as string "1", "2", ...
        page_label = doc.metadata.get("page_label", str(i + 1))
        try:
            page_num = int(page_label)
        except ValueError:
            page_num = i + 1

        doc.metadata["page_number"] = page_num
        doc.metadata["source"] = f"Cyber Ireland 2022 Report - Page {page_num}"

    return raw_docs


# =============================================================================
# STEP 2 - Parse Markdown Tables -> tables.json
# =============================================================================

def _parse_md_tables_from_page(markdown_text: str, page_num: int) -> list:
    """
    Extract markdown tables ( | col | col | rows ) from a page's markdown.
    Finds the nearest heading/title line above each table.
    """
    tables   = []
    lines    = markdown_text.split("\n")
    n        = len(lines)
    i        = 0

    while i < n:
        line = lines[i].strip()

        # Table row: must have at least 2 pipe characters
        if line.startswith("|") and line.count("|") >= 2:

            # Search backwards for nearest non-pipe, non-empty line = title
            title = f"Table (Page {page_num})"
            for back in range(i - 1, max(i - 6, -1), -1):
                candidate = lines[back].strip()
                if candidate and "|" not in candidate and not candidate.startswith("#"):
                    title = re.sub(r"[*_#`]", "", candidate).strip()
                    break
                if candidate.startswith("#"):
                    title = re.sub(r"[*_#`]", "", candidate).strip()
                    break

            # Collect all consecutive table lines
            tbl_lines = []
            while i < n and "|" in lines[i]:
                tbl_lines.append(lines[i].strip())
                i += 1

            if len(tbl_lines) < 2:
                continue

            # Row 0 = headers
            headers = [h.strip() for h in tbl_lines[0].split("|") if h.strip()]

            # Row 1 might be separator (---|---)
            row_start = 1
            if len(tbl_lines) > 1:
                sep = tbl_lines[1].replace("|", "").replace("-", "").replace(":", "").strip()
                if sep == "":
                    row_start = 2

            rows = []
            for rl in tbl_lines[row_start:]:
                row = [c.strip() for c in rl.split("|") if c.strip()]
                if row:
                    rows.append(row)

            if headers and rows:
                tables.append({
                    "page":    page_num,
                    "title":   title,
                    "headers": headers,
                    "rows":    rows,
                })
        else:
            i += 1

    return tables


def extract_tables_from_docs(docs: list) -> list:
    """Run markdown table parser over every page document."""
    all_tables = []
    for doc in docs:
        page_num = doc.metadata.get("page_number", 0)
        tables   = _parse_md_tables_from_page(doc.text, page_num)
        all_tables.extend(tables)
    return all_tables


# =============================================================================
# STEP 3 - Embed + Store in ChromaDB
# =============================================================================

def build_chroma_store():
    """Reset ChromaDB collection and return LlamaIndex vector store + storage context."""
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


# =============================================================================
# STEP 4 - Validation smoke-test
# =============================================================================

def validate(index, embed_model):
    Settings.embed_model = embed_model
    retriever = index.as_retriever(similarity_top_k=3)
    nodes     = retriever.retrieve("total number of jobs cybersecurity Ireland")

    print("\n  Smoke-test - 'total number of jobs cybersecurity Ireland'")
    for i, node in enumerate(nodes, 1):
        pg      = node.metadata.get("page_number", "?")
        score   = round(node.score, 4) if node.score else "n/a"
        preview = node.text[:160].replace("\n", " ")
        print(f"    [{i}] Page {pg} (score={score}): {preview}...")


# =============================================================================
# MAIN
# =============================================================================

def ingest(pdf_path: Path) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  InsightForge.AI - ETL Pipeline  [LlamaParse Edition]")
    print(f"{sep}")
    print(f"  PDF    : {pdf_path.name}")
    print(f"  Parser : LlamaParse (result_type=markdown)")
    print(f"  Embed  : {EMBED_MODEL}")
    print(f"  Chunk  : {CHUNK_SIZE} tokens  |  Overlap: {CHUNK_OVERLAP} tokens")
    print(f"  top_k  : 8  (no reranker - single PDF corpus)")
    print(f"{sep}\n")

    # 1. Parse PDF with LlamaParse
    print("[1/4] Parsing PDF with LlamaParse ...")
    documents = parse_pdf_with_llamaparse(pdf_path)
    print(f"      -> {len(documents)} pages parsed as markdown")

    # 2. Extract tables from markdown output
    print("[2/4] Extracting tables from markdown ...")
    tables = extract_tables_from_docs(documents)
    TABLES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLES_OUT, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"      -> {len(tables)} tables extracted -> {TABLES_OUT.name}")

    # 3. Embed + Store
    print("[3/4] Chunking, embedding, storing in ChromaDB ...")
    embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=OPENAI_API_KEY)

    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
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
    print(f"  ETL Complete - LlamaParse Edition")
    print(f"{sep}")
    print(f"  Pages parsed   : {len(documents)}")
    print(f"  Tables saved   : {len(tables)}")
    print(f"  ChromaDB       : {CHROMA_DIR}")
    print(f"  Tables JSON    : {TABLES_OUT}")
    print(f"{sep}\n")


if __name__ == "__main__":
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)
    ingest(PDF_PATH)
