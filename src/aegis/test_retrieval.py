"""
Aegis: smoke test for the retrieval index.

Loads the Chroma collection built by build_index.py and runs three sample
queries against it. Prints the top retrieved chunks and the page each came
from, so a human can sanity-check that retrieval is returning sensible
passages of the Act.

This is not a quality benchmark. The Week 8 evaluation harness does that
properly against a hand-labelled golden set. This script just confirms
the pipeline is connected end to end.

Copyright (c) 2026 Noble Chidera Onyema. All Rights Reserved.
"""

from pathlib import Path
import sys
import textwrap

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "ai_act_v1"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Three queries chosen to cover different parts of the Act.
SAMPLE_QUERIES = [
    "What counts as a high-risk AI system?",
    "What are the transparency obligations for AI systems that interact with humans?",
    "Which AI practices are prohibited under the Act?",
]

TOP_K = 3
WRAP_WIDTH = 90


def main() -> int:
    if not CHROMA_DIR.exists():
        print(f"ERROR: no Chroma DB at {CHROMA_DIR}. Run build_index.py first.")
        return 1

    print(f"Loading Chroma collection '{COLLECTION_NAME}' from {CHROMA_DIR}...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"  Collection has {chroma_collection.count()} chunks.")
    print()

    print(f"Loading embedding model {EMBED_MODEL_NAME}...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.llm = None
    print()

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    retriever = index.as_retriever(similarity_top_k=TOP_K)

    for query in SAMPLE_QUERIES:
        print("=" * WRAP_WIDTH)
        print(f"QUERY: {query}")
        print("=" * WRAP_WIDTH)
        results = retriever.retrieve(query)
        for rank, node in enumerate(results, start=1):
            page = node.metadata.get("page", "?")
            score = node.score if node.score is not None else 0.0
            snippet = node.text.replace("\n", " ").strip()
            snippet = textwrap.shorten(snippet, width=400, placeholder=" [...]")
            wrapped = textwrap.fill(snippet, width=WRAP_WIDTH)
            print(f"\n[{rank}] page {page}, similarity {score:.3f}")
            print(wrapped)
        print()

    print("Retrieval pipeline is working end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())