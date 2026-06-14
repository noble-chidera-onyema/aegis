"""
Aegis index builder, v3 (Article- and Annex-aware).

Reads ai_act.pdf with pdfplumber, detects every "Article N" AND "ANNEX N"
header, splits on those boundaries, and tags each chunk with:
  - provision: "Article 14", "Annex III", or "Recitals"
  - provision_start_page: the page where that provision begins
  - page: the real page this chunk's own text sits on

Why both pages: citations should name the provision and the page where it
begins (the page a reader flips to), while the chunk's own page is kept for
audit. v1 had no Annex boundaries at all, so Annex III text was glued onto the
last Article's chunks with no Annex metadata.

Run from the project root:
    python src/aegis/build_index.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import chromadb
import pdfplumber
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --- Configuration ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ai_act_v1"

SOURCE_PDFS = ["ai_act.pdf"]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SUB_CHUNK_SIZE = 1500
SUB_CHUNK_OVERLAP = 200

ARTICLE_HEADER = re.compile(r"^Article\s+(\d+)\s*$", re.MULTILINE)
ANNEX_HEADER = re.compile(r"^ANNEX\s+([IVXLC]+)\s*$", re.MULTILINE)


def extract_full_text_with_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text page by page. Returns list of (page_number, text)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def build_char_to_page(pages: list[tuple[int, str]]):
    """
    Concatenate pages into one text and return (full_text, char_to_page),
    where char_to_page(char_index) gives the real page for that position.
    """
    full_text = ""
    page_offsets = []
    for page_num, text in pages:
        page_offsets.append((len(full_text), page_num))
        full_text += text + "\n"

    def char_to_page(char_idx: int) -> int:
        page = page_offsets[0][1] if page_offsets else 1
        for offset, pnum in page_offsets:
            if offset <= char_idx:
                page = pnum
            else:
                break
        return page

    return full_text, char_to_page


def find_boundaries(full_text: str, char_to_page) -> list[dict]:
    """
    Detect every Article and Annex header. Returns boundaries sorted by
    position, each with: provision ("Article 14" / "Annex III"),
    article_number (int, 0 for annexes), start_char, start_page.
    """
    boundaries = []
    for match in ARTICLE_HEADER.finditer(full_text):
        num = int(match.group(1))
        boundaries.append({
            "provision": f"Article {num}",
            "article_number": num,
            "start_char": match.start(),
            "start_page": char_to_page(match.start()),
        })
    for match in ANNEX_HEADER.finditer(full_text):
        numeral = match.group(1)
        boundaries.append({
            "provision": f"Annex {numeral}",
            "article_number": 0,
            "start_char": match.start(),
            "start_page": char_to_page(match.start()),
        })
    boundaries.sort(key=lambda b: b["start_char"])
    return boundaries


def _page_for_subchunk(sub_text: str, search_from: int, full_text: str,
                       char_to_page, fallback_page: int) -> tuple[int, int]:
    """
    Locate sub_text in full_text from search_from; return (page, new_cursor).
    Falls back to fallback_page if not found.
    """
    idx = full_text.find(sub_text[:200], search_from)
    if idx == -1:
        anchor = sub_text.strip()[:40]
        if anchor:
            idx = full_text.find(anchor, search_from)
    if idx == -1:
        return fallback_page, search_from
    return char_to_page(idx), idx + 1


def build_provision_chunks(full_text: str, boundaries: list[dict],
                           char_to_page) -> list[dict]:
    """
    Split full_text into provision-bounded chunks (Articles and Annexes).
    Sub-chunks of long provisions get their own real page; every chunk also
    carries the provision's start page.
    """
    chunks = []
    splitter = SentenceSplitter(
        chunk_size=SUB_CHUNK_SIZE,
        chunk_overlap=SUB_CHUNK_OVERLAP,
    )

    for i, b in enumerate(boundaries):
        start = b["start_char"]
        end = boundaries[i + 1]["start_char"] if i + 1 < len(boundaries) else len(full_text)
        text = full_text[start:end].strip()
        if not text:
            continue

        base = {
            "provision": b["provision"],
            "article_number": b["article_number"],
            "provision_start_page": b["start_page"],
        }

        if len(text) <= SUB_CHUNK_SIZE:
            chunks.append({**base, "text": text, "page": b["start_page"],
                           "sub_chunk_index": 0})
        else:
            sub_texts = splitter.split_text(text)
            cursor = start
            for j, sub_text in enumerate(sub_texts):
                page, cursor = _page_for_subchunk(
                    sub_text, cursor, full_text, char_to_page, b["start_page"]
                )
                chunks.append({**base, "text": sub_text, "page": page,
                               "sub_chunk_index": j})

    return chunks


def build_preamble_chunks(full_text: str, first_boundary_start: int,
                          char_to_page) -> list[dict]:
    """
    Text before the first Article/Annex (cover, recitals) becomes chunks
    tagged provision="Recitals". Each gets its real page.
    """
    preamble = full_text[:first_boundary_start]
    if not preamble.strip():
        return []
    splitter = SentenceSplitter(
        chunk_size=SUB_CHUNK_SIZE,
        chunk_overlap=SUB_CHUNK_OVERLAP,
    )
    sub_texts = splitter.split_text(preamble)
    chunks = []
    cursor = 0
    for j, t in enumerate(sub_texts):
        page, cursor = _page_for_subchunk(t, cursor, full_text, char_to_page, 1)
        chunks.append({
            "text": t,
            "provision": "Recitals",
            "article_number": 0,
            "provision_start_page": page,  # for recitals, cite the excerpt's page
            "page": page,
            "sub_chunk_index": j,
        })
    return chunks


def main() -> int:
    print("Aegis index builder v3 (Article- and Annex-aware)")
    print(f"  Data directory : {DATA_DIR}")
    print(f"  Chroma DB path : {CHROMA_DB_PATH}")
    print(f"  Collection     : {COLLECTION_NAME}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print()

    print("[1/5] Extracting text with pdfplumber...")
    all_documents = []
    for pdf_name in SOURCE_PDFS:
        pdf_path = DATA_DIR / pdf_name
        if not pdf_path.exists():
            print(f"      WARNING: {pdf_path} not found, skipping")
            continue
        pages = extract_full_text_with_pages(pdf_path)
        print(f"      {pdf_name}: {len(pages)} pages with text")

        full_text, char_to_page = build_char_to_page(pages)

        print(f"[2/5] Detecting Article and Annex boundaries in {pdf_name}...")
        boundaries = find_boundaries(full_text, char_to_page)
        articles = [b for b in boundaries if b["provision"].startswith("Article")]
        annexes = [b for b in boundaries if b["provision"].startswith("Annex")]
        print(f"      Articles found: {len(articles)}")
        print(f"      Annexes found : {len(annexes)} -> {[b['provision'] for b in annexes]}")
        if annexes:
            print(f"      Annex start pages: {[(b['provision'], b['start_page']) for b in annexes]}")
        if not annexes:
            print("      WARNING: no ANNEX headers detected. Check the regex against the PDF's annex heading format.")

        if not boundaries:
            print(f"      WARNING: no boundaries detected. Chunking as preamble only.")
            preamble_chunks = build_preamble_chunks(full_text, len(full_text), char_to_page)
            all_chunks = preamble_chunks
        else:
            print(f"[3/5] Building provision-bounded chunks...")
            preamble_chunks = build_preamble_chunks(full_text, boundaries[0]["start_char"], char_to_page)
            provision_chunks = build_provision_chunks(full_text, boundaries, char_to_page)
            all_chunks = preamble_chunks + provision_chunks
            print(f"      Preamble chunks : {len(preamble_chunks)}")
            print(f"      Provision chunks: {len(provision_chunks)}")
            print(f"      Total chunks    : {len(all_chunks)}")

        page1 = sum(1 for c in all_chunks if c["page"] == 1)
        print(f"      Chunks on page 1: {page1} (a small number is expected)")

        for c in all_chunks:
            all_documents.append(Document(
                text=c["text"],
                metadata={
                    "source": pdf_name,
                    "provision": c["provision"],
                    "article_number": c["article_number"],
                    "provision_start_page": c["provision_start_page"],
                    "page": c["page"],
                    "sub_chunk_index": c["sub_chunk_index"],
                },
            ))

    print()
    print(f"[4/5] Loading embedding model (cached after first run)...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    print(f"      Loaded {EMBEDDING_MODEL}.")

    print(f"[5/5] Embedding chunks and writing to Chroma. Takes 1 to 3 minutes...")
    db = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        db.delete_collection(COLLECTION_NAME)
        print(f"      Dropped previous collection.")
    except Exception:
        pass

    collection = db.create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    start = time.time()
    VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    elapsed = time.time() - start

    print()
    print(f"Done. {collection.count()} chunks indexed in collection '{COLLECTION_NAME}'.")
    print(f"Elapsed: {elapsed:.1f} seconds.")
    print(f"On-disk location: {CHROMA_DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
