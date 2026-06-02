"""
Aegis index builder (Article-aware version).

Reads ai_act.pdf with pdfplumber, then performs Article-aware text splitting:
detects every "Article N" header in the document, splits the text on those
boundaries, and tags each resulting chunk with structured Article metadata.

This architecture replaces text-search-based citation lookup with metadata
filtering. Each chunk knows which Article it belongs to via metadata, so
finding "the text of Article 9" is a metadata query, not a regex search
that depends on chunk-boundary luck.

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

# Sub-chunking inside long Articles: only kicks in when an Article exceeds
# this many characters. Most Articles are shorter than this and stay in
# one chunk along with their header.
SUB_CHUNK_SIZE = 1500
SUB_CHUNK_OVERLAP = 200

# Matches an Article header line: "Article 9" at the start of a line.
# The (?=\n) lookahead ensures we match a standalone header line, not
# a cross-reference embedded in a sentence.
ARTICLE_HEADER = re.compile(r"^Article\s+(\d+)\s*$", re.MULTILINE)


def extract_full_text_with_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text page by page. Returns list of (page_number, text)."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def find_article_boundaries(pages: list[tuple[int, str]]) -> list[dict]:
    """
    Detect every Article header in the document and return a list of
    Article boundaries.

    Returns: list of dicts with keys: article_number, start_page,
    start_char_in_concatenated, header_text.
    """
    # Concatenate all pages into one big text with page markers we can
    # use to track which page each character lives on.
    full_text = ""
    page_offsets = []  # (start_char_index, page_number) pairs
    for page_num, text in pages:
        page_offsets.append((len(full_text), page_num))
        full_text += text + "\n"

    def char_to_page(char_idx: int) -> int:
        """Map a character index in full_text back to a page number."""
        page = page_offsets[0][1]
        for offset, pnum in page_offsets:
            if offset <= char_idx:
                page = pnum
            else:
                break
        return page

    boundaries = []
    for match in ARTICLE_HEADER.finditer(full_text):
        article_num = int(match.group(1))
        start = match.start()
        boundaries.append({
            "article_number": article_num,
            "start_char": start,
            "start_page": char_to_page(start),
            "header_text": match.group(0),
        })

    return boundaries, full_text


def build_article_chunks(full_text: str, boundaries: list[dict]) -> list[dict]:
    """
    Split the full text into Article-bounded chunks.

    Each chunk represents one Article (or a portion of one long Article).
    Returns: list of dicts with text, article_number, start_page, sub_chunk_index.
    """
    chunks = []
    splitter = SentenceSplitter(
        chunk_size=SUB_CHUNK_SIZE,
        chunk_overlap=SUB_CHUNK_OVERLAP,
    )

    for i, b in enumerate(boundaries):
        start = b["start_char"]
        end = boundaries[i + 1]["start_char"] if i + 1 < len(boundaries) else len(full_text)
        article_text = full_text[start:end].strip()

        if not article_text:
            continue

        if len(article_text) <= SUB_CHUNK_SIZE:
            # Short Article fits in one chunk along with its header.
            chunks.append({
                "text": article_text,
                "article_number": b["article_number"],
                "start_page": b["start_page"],
                "sub_chunk_index": 0,
            })
        else:
            # Long Article: split into sub-chunks. Every sub-chunk keeps
            # the same article_number so metadata lookup still works.
            sub_texts = splitter.split_text(article_text)
            for j, sub_text in enumerate(sub_texts):
                chunks.append({
                    "text": sub_text,
                    "article_number": b["article_number"],
                    "start_page": b["start_page"],
                    "sub_chunk_index": j,
                })

    return chunks


def build_preamble_chunks(full_text: str, first_article_start: int) -> list[dict]:
    """
    The text before the first Article (cover page, recitals, table of contents)
    becomes its own set of chunks tagged article_number=0. This keeps the recital
    text searchable for Q&A but distinguishable from operative Articles.
    """
    preamble = full_text[:first_article_start]
    if not preamble.strip():
        return []
    splitter = SentenceSplitter(
        chunk_size=SUB_CHUNK_SIZE,
        chunk_overlap=SUB_CHUNK_OVERLAP,
    )
    sub_texts = splitter.split_text(preamble)
    return [
        {
            "text": t,
            "article_number": 0,  # 0 = preamble/recitals
            "start_page": 1,
            "sub_chunk_index": j,
        }
        for j, t in enumerate(sub_texts)
    ]


def main() -> int:
    print("Aegis index builder (Article-aware)")
    print(f"  Data directory : {DATA_DIR}")
    print(f"  Chroma DB path : {CHROMA_DB_PATH}")
    print(f"  Collection     : {COLLECTION_NAME}")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print()

    # Step 1: extract text from PDFs.
    print("[1/5] Extracting text with pdfplumber...")
    all_documents = []
    for pdf_name in SOURCE_PDFS:
        pdf_path = DATA_DIR / pdf_name
        if not pdf_path.exists():
            print(f"      WARNING: {pdf_path} not found, skipping")
            continue
        pages = extract_full_text_with_pages(pdf_path)
        print(f"      {pdf_name}: {len(pages)} pages with text")

        # Step 2: detect Article boundaries.
        print(f"[2/5] Detecting Article boundaries in {pdf_name}...")
        boundaries, full_text = find_article_boundaries(pages)
        article_nums = sorted(set(b["article_number"] for b in boundaries))
        print(f"      Found {len(boundaries)} Article header(s).")
        print(f"      Article numbers detected: {article_nums[:5]}...{article_nums[-5:]}")

        if not boundaries:
            print(f"      WARNING: no Articles detected. Document will be chunked as preamble only.")
            preamble_chunks = build_preamble_chunks(full_text, len(full_text))
            for c in preamble_chunks:
                all_documents.append(Document(
                    text=c["text"],
                    metadata={
                        "source": pdf_name,
                        "article_number": c["article_number"],
                        "page": c["start_page"],
                        "sub_chunk_index": c["sub_chunk_index"],
                    },
                ))
            continue

        # Step 3: build chunks.
        print(f"[3/5] Building Article-bounded chunks...")
        preamble_chunks = build_preamble_chunks(full_text, boundaries[0]["start_char"])
        article_chunks = build_article_chunks(full_text, boundaries)
        all_chunks = preamble_chunks + article_chunks
        print(f"      Preamble chunks: {len(preamble_chunks)}")
        print(f"      Article chunks : {len(article_chunks)}")
        print(f"      Total chunks   : {len(all_chunks)}")

        for c in all_chunks:
            all_documents.append(Document(
                text=c["text"],
                metadata={
                    "source": pdf_name,
                    "article_number": c["article_number"],
                    "page": c["start_page"],
                    "sub_chunk_index": c["sub_chunk_index"],
                },
            ))

    print()

    # Step 4: load the embedding model.
    print(f"[4/5] Loading embedding model (cached after first run)...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    print(f"      Loaded {EMBEDDING_MODEL}.")

    # Step 5: re-create Chroma collection and index.
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