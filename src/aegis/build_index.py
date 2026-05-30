"""
Aegis: build the retrieval index for the EU AI Act.

Pipeline:
  1. Read data/ai_act.pdf with pypdf, one Document per page.
  2. Split each page into chunks (~800 chars, 100 overlap).
  3. Embed every chunk with sentence-transformers/all-MiniLM-L6-v2.
  4. Write the chunks and embeddings to a Chroma collection at ./chroma_db/.

Run once per change to the source PDF. The chroma_db/ folder is gitignored;
it rebuilds on any machine from the same PDF.

Copyright (c) 2026 Noble Chidera Onyema. All Rights Reserved.
"""

from pathlib import Path
import sys

import chromadb
import pypdf
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# Project root is two folders up from src/aegis/build_index.py.
ROOT = Path(__file__).resolve().parent.parent.parent
PDF_PATH = ROOT / "data" / "ai_act.pdf"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "ai_act_v1"

# Chunking config. 800 chars is roughly one paragraph of dense legal text.
# A 100-char overlap means adjacent chunks share their boundary, which helps
# retrieval when a relevant sentence happens to straddle the boundary.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Embedding model. all-MiniLM-L6-v2 produces 384-dimensional vectors.
# Small enough to run on CPU. Standard choice for short technical English text.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> int:
    print("Aegis index builder")
    print(f"  PDF source     : {PDF_PATH}")
    print(f"  Chroma DB path : {CHROMA_DIR}")
    print(f"  Collection     : {COLLECTION_NAME}")
    print(f"  Embedding model: {EMBED_MODEL_NAME}")
    print()

    if not PDF_PATH.exists():
        print(f"ERROR: cannot find {PDF_PATH}. Are you running from the project root?")
        return 1

    # Step 1: read the PDF, one Document per page, page number kept as metadata.
    print("[1/4] Reading PDF...")
    reader = pypdf.PdfReader(str(PDF_PATH))
    total_pages = len(reader.pages)
    documents = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                Document(
                    text=text,
                    metadata={"source": "ai_act.pdf", "page": i},
                )
            )
    print(f"      Read {total_pages} pages, kept {len(documents)} non-empty ones.")
    print()

    # Step 2: configure the embedding model.
    # First run downloads the model (~80MB) to the local Hugging Face cache.
    print("[2/4] Loading embedding model (~80MB on first run, cached after)...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    # No LLM is needed for indexing. Set to None so LlamaIndex does not try to
    # load a default (which would fail without an OpenAI key).
    Settings.llm = None
    print(f"      Loaded {EMBED_MODEL_NAME}.")
    print()

    # Step 3: configure the chunker.
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"[3/4] Chunker ready: {CHUNK_SIZE} chars per chunk, {CHUNK_OVERLAP} overlap.")
    print()

    # Step 4: connect to (or create) the persistent Chroma collection,
    # then run the indexing pipeline.
    print("[4/4] Embedding chunks and writing to Chroma. Takes 1 to 3 minutes...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe and recreate the collection so re-runs are idempotent. Without this,
    # rerunning the script would duplicate every chunk.
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    chroma_collection = chroma_client.create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    final_count = chroma_collection.count()
    print()
    print(f"Done. {final_count} chunks indexed in collection '{COLLECTION_NAME}'.")
    print(f"On-disk location: {CHROMA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())