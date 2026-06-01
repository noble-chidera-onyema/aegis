"""
Aegis risk classifier: take a plain-language description of an AI system,
return a structured classification into one of four EU AI Act risk tiers
(prohibited, high-risk, limited-risk, minimal-risk) with reasoning and
citations to specific articles or Annex III categories.

Designed as a function module first, with a CLI entry point at the bottom
for development testing. The Week 6 Streamlit UI will import
classify_system() directly.

Run from the project root for a CLI demonstration:
    python src/aegis/classify.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import chromadb
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# --- Configuration ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ai_act_v1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 8  # More context than Q&A because classification needs broader coverage.
WRAP_WIDTH = 88


# --- The four tiers ---------------------------------------------------------

Tier = Literal["prohibited", "high-risk", "limited-risk", "minimal-risk"]
VALID_TIERS: tuple[Tier, ...] = ("prohibited", "high-risk", "limited-risk", "minimal-risk")


@dataclass
class Classification:
    """The structured result returned to callers and the UI."""
    tier: Tier
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    citations: list[str]
    needs_human_review: bool
    raw_response: str  # the unparsed model output, for debugging and audit


# --- System prompt ----------------------------------------------------------

SYSTEM_PROMPT = """You are a compliance research assistant for the EU AI Act. Your job is to classify a described AI system into one of four risk tiers and explain why, using only the passages provided from the Act itself.

The four tiers:
- "prohibited": practices banned under Article 5 (subliminal manipulation, exploitation of vulnerabilities, social scoring, untargeted scraping of facial images for databases, emotion inference in workplace or education, biometric categorisation by sensitive attributes, certain real-time biometric identification in public spaces).
- "high-risk": systems listed in Annex III (biometrics, critical infrastructure, education, employment, essential services, law enforcement, migration, justice, democratic processes) and certain safety components under Annex I.
- "limited-risk": systems subject to transparency obligations under Article 50 (chatbots, deepfakes, AI-generated content, emotion recognition outside prohibited contexts).
- "minimal-risk": everything else. Most AI systems fall here.

Output rules:
1. Respond with ONLY a JSON object. No preamble. No closing remarks. No markdown fences.
2. The JSON must have exactly these keys: tier, confidence, reasoning, citations, needs_human_review.
3. tier must be one of: "prohibited", "high-risk", "limited-risk", "minimal-risk".
4. confidence must be one of: "high", "medium", "low".
5. reasoning is a short paragraph (3 to 6 sentences) explaining the classification, grounded in the passages provided.
6. citations is a list of strings, each naming the specific Article or Annex section, with a page number where the passage came from. Example: "Article 5(1)(a), page 51" or "Annex III, point 4(a), page 130".
7. needs_human_review is a boolean. Set it to true if any of the following apply: the description is ambiguous, the description falls near a tier boundary, the retrieved passages do not strongly support the classification, or the system might also implicate transparency obligations under Article 50 in addition to its primary tier.
8. If the retrieved passages do not cover the question, set tier to "minimal-risk" with confidence "low" and needs_human_review to true, and say so in reasoning.

This is decision-support for compliance officers. Be cautious. When in doubt, classify higher and flag for human review.
"""


# --- Index loading ----------------------------------------------------------

def load_index() -> VectorStoreIndex:
    """Open the Chroma collection built in Week 2."""
    db = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = db.get_collection(COLLECTION_NAME)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def retrieve_chunks(index: VectorStoreIndex, description: str, top_k: int = TOP_K):
    """Return the top-k most relevant chunks for a system description."""
    # Augment the retrieval query with classification-relevant terms so we
    # pull Article 5 (prohibited) and Annex III (high-risk) chunks even when
    # the description doesn't use those words.
    retrieval_query = (
        f"{description} prohibited practices high-risk AI systems Annex III "
        f"transparency obligations Article 5 Article 50"
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(retrieval_query)


def build_user_prompt(description: str, retrieved_nodes) -> str:
    """Format description + retrieved chunks into the user message."""
    chunks_text = []
    for i, node in enumerate(retrieved_nodes, start=1):
        page = node.metadata.get("page", "?")
        text = node.text.replace("\n", " ").strip()
        chunks_text.append(f"[Passage {i}, page {page}]\n{text}")

    chunks_block = "\n\n".join(chunks_text)

    return (
        f"AI system description:\n{description}\n\n"
        f"Relevant passages from the EU AI Act:\n\n"
        f"{chunks_block}\n\n"
        f"Classify this AI system. Respond with only the JSON object."
    )


# --- The public function ---------------------------------------------------

def classify_system(description: str, client: Groq | None = None,
                    index: VectorStoreIndex | None = None) -> Classification:
    """
    Classify an AI system description into one of four EU AI Act risk tiers.

    Args:
        description: plain-language description of the AI system.
        client: an existing Groq client. Created from env if not supplied.
        index: an existing VectorStoreIndex. Loaded from disk if not supplied.

    Returns:
        Classification with tier, confidence, reasoning, citations, and a
        needs_human_review flag.
    """
    if client is None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        client = Groq(api_key=api_key)

    if index is None:
        index = load_index()

    chunks = retrieve_chunks(index, description)
    user_prompt = build_user_prompt(description, chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,  # zero for maximum consistency on classification.
        max_tokens=800,
        response_format={"type": "json_object"},  # Groq supports JSON mode.
    )
    raw = response.choices[0].message.content

    return _parse_response(raw)


def _parse_response(raw: str) -> Classification:
    """Parse and validate the model's JSON response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Defensive fallback. JSON mode should prevent this but never trust the model.
        return Classification(
            tier="minimal-risk",
            confidence="low",
            reasoning=f"Could not parse model response as JSON: {exc}. Raw response preserved below.",
            citations=[],
            needs_human_review=True,
            raw_response=raw,
        )

    tier = data.get("tier", "minimal-risk")
    if tier not in VALID_TIERS:
        # Model returned a tier name we don't recognise. Fail safe.
        return Classification(
            tier="minimal-risk",
            confidence="low",
            reasoning=f"Model returned unrecognised tier '{tier}'. Falling back to minimal-risk and flagging for review.",
            citations=[],
            needs_human_review=True,
            raw_response=raw,
        )

    return Classification(
        tier=tier,
        confidence=data.get("confidence", "low"),
        reasoning=data.get("reasoning", ""),
        citations=data.get("citations", []),
        needs_human_review=bool(data.get("needs_human_review", True)),
        raw_response=raw,
    )


# --- CLI entry point --------------------------------------------------------

# Four test descriptions, one targeted at each tier, used during development.
# Not the formal evaluation set. Week 8 builds 30 to 50 hand-labelled examples.
TEST_DESCRIPTIONS = [
    {
        "label": "1. Hiring CV screener",
        "expected_tier": "high-risk",
        "description": (
            "We are a 40-person company in Dublin. We use an AI tool that "
            "reads incoming CVs and ranks candidates by predicted fit with "
            "the role. The shortlist goes to a human recruiter who makes "
            "the final interview decisions."
        ),
    },
    {
        "label": "2. Customer service chatbot",
        "expected_tier": "limited-risk",
        "description": (
            "We run an Irish e-commerce site. We have an AI chatbot that "
            "answers customer questions about orders, returns, and product "
            "availability. The chatbot is clearly labelled as automated."
        ),
    },
    {
        "label": "3. Social scoring system",
        "expected_tier": "prohibited",
        "description": (
            "We are building an AI system that scores citizens based on "
            "their social media activity, payment history, and public "
            "behaviour records, and uses the score to decide whether they "
            "get access to certain public services."
        ),
    },
    {
        "label": "4. Internal spam filter",
        "expected_tier": "minimal-risk",
        "description": (
            "Our 12-person law firm uses an AI spam filter on the company "
            "email server. It classifies incoming email as spam or not "
            "spam. No external users are affected."
        ),
    },
]


def _print_classification(c: Classification) -> None:
    """Format a Classification for the terminal."""
    print(f"\n  Tier:                  {c.tier}")
    print(f"  Confidence:            {c.confidence}")
    print(f"  Needs human review:    {c.needs_human_review}")
    print(f"\n  Reasoning:")
    print(textwrap.fill(c.reasoning, width=WRAP_WIDTH,
                        initial_indent="    ", subsequent_indent="    "))
    print(f"\n  Citations:")
    for cite in c.citations:
        print(f"    - {cite}")
    print()


def main() -> int:
    print("Loading Chroma index and Groq client...")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set. Add it to your .env file.")
        return 1

    client = Groq(api_key=api_key)
    index = load_index()
    print("Ready.\n")

    for case in TEST_DESCRIPTIONS:
        print("=" * WRAP_WIDTH)
        print(case["label"])
        print(f"Expected tier: {case['expected_tier']}")
        print("=" * WRAP_WIDTH)
        print()
        print("Description:")
        print(textwrap.fill(case["description"], width=WRAP_WIDTH,
                            initial_indent="  ", subsequent_indent="  "))

        result = classify_system(case["description"], client=client, index=index)
        _print_classification(result)

        match = "match" if result.tier == case["expected_tier"] else "MISMATCH"
        print(f"  Expected: {case['expected_tier']}   Got: {result.tier}   ({match})\n")

    print("Risk classifier is working end to end.")
    print("This is decision-support, not legal advice. Verify with qualified counsel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())