"""
Aegis risk classifier, v4.

Changes from v3, each driven by the v3 evaluation results:
1. Limited-risk step rebalanced. v3 told the model not to use limited-risk as a
   compromise; it over-corrected and pushed genuine Article 50 cases (chatbots,
   deepfakes) into minimal-risk and high-risk, dropping limited-risk recall to
   2 of 7. v4 states the Article 50 triggers as positive signals: if a system
   is a chatbot or generates synthetic content and is not caught earlier, it IS
   limited-risk. The higher tier still wins when high-risk also applies.
2. Review flag from structural signals. v3 showed the model rates almost every
   case "high" confidence, so a confidence-only floor rarely fired. v4 also
   fires the flag when the model's own reasoning text names an exception, a
   boundary, or a contested or interpretive judgement. Computed in code, not
   dependent on the model setting the flag.

Kept from v3: Annex-aware indexing, the provision-start-page citation
convention, the Article 5 gates and exceptions, the Annex III 5(b) fraud
carve-out. Those improved citation accuracy and are unchanged.

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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = 8
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
    raw_response: str


# --- System prompt ----------------------------------------------------------

SYSTEM_PROMPT = """You are a compliance research assistant for the EU AI Act. Classify a described AI system into one of four risk tiers using only the passages provided from the Act.

Apply this decision procedure IN ORDER:

STEP 1, Article 5 (prohibited). Check each Article 5 practice. Two gates matter:
- The manipulation and exploitation prohibitions (5(1)(a), 5(1)(b)) require BOTH material distortion of behaviour AND significant harm. Ordinary commercial persuasion, engagement features, or upselling without significant harm do NOT meet the gate.
- Several prohibitions carry exceptions. Emotion inference in the workplace is NOT prohibited when for medical or safety reasons (5(1)(f)). Real-time remote biometric identification for law enforcement under the listed exceptions in 5(1)(h) is NOT prohibited; it is regulated as high-risk instead.
If a prohibition fully applies with its gates met and no exception, the tier is "prohibited".

STEP 2, high-risk. Check Annex III use cases (biometrics, critical infrastructure, education, employment, essential services, law enforcement, migration, justice, democratic processes) and Article 6(1) safety components of regulated products (e.g. medical devices). Notes:
- Annex III point 5(b) creditworthiness EXCLUDES AI used to detect financial fraud. Fraud detection is not high-risk under 5(b).
- A human reviewing or having override does NOT remove high-risk status when the system materially influences the decision (evaluating candidates, grading students, proctoring exams).

STEP 3, limited-risk (Article 50 transparency). Assign limited-risk when the system has a transparency trigger under Article 50 but is NOT high-risk or prohibited. The triggers, assign limited-risk if ANY apply:
- It interacts directly with people (a chatbot, a conversational agent, a virtual assistant). A customer-facing chatbot is the textbook limited-risk case.
- It generates or manipulates synthetic audio, image, video, or text content (including deep fakes and AI-written content shown to people).
- It is an emotion-recognition or biometric-categorisation system used outside the prohibited contexts of Article 5.
Apply these notes:
- These triggers are POSITIVE signals. If a system is a chatbot or generates synthetic content and is not caught by Step 1 or Step 2, it IS limited-risk. Do not push it down to minimal-risk for lacking some further condition; the trigger itself is sufficient.
- A system can be high-risk under Step 2 AND owe Article 50 transparency. When both apply, the tier is high-risk (the higher tier wins) and the transparency duty is noted in reasoning, not used to downgrade.
- The only things that defeat a Step 3 trigger: the system merely assists standard editing (spell-check, grammar, transcription) or processes only the user's own content for that user. Those are minimal-risk.

STEP 4, otherwise "minimal-risk". Most AI systems land here. Internal tools, logistics, forecasting, quality control, and recommendation systems with no Annex III use are minimal-risk.

Citation rules:
- Each passage is labelled with its provision (e.g. "Article 14" or "Annex III") and the page where that provision BEGINS. Cite the provision and that beginning page. Example: "Annex III, point 4(b), page 127" or "Article 50(1), page 82".
- Never cite passage numbers (never write "Passage 3"). Never write "page not specified". If you rely on a recital, cite "Recitals, page N" using the page shown on that passage.

Output rules:
1. Respond with ONLY a JSON object. No preamble. No markdown fences.
2. Keys: tier, confidence, reasoning, citations, needs_human_review.
3. tier: one of "prohibited", "high-risk", "limited-risk", "minimal-risk".
4. confidence: one of "high", "medium", "low". Use "high" only when the classification is clear-cut under the decision procedure; use "medium" or "low" whenever the case sits near a tier boundary or depends on an exception.
5. reasoning: 3 to 6 sentences, grounded in the provided passages.
6. citations: list of strings per the citation rules above.
7. needs_human_review: true if the description is ambiguous, near a tier boundary, depends on an exception or a contested reading, or the passages weakly support the classification.
8. If the passages do not cover the question, use tier "minimal-risk", confidence "low", needs_human_review true, and say so in reasoning.

This is decision-support for compliance officers, not legal advice. Be cautious. When genuinely in doubt between two tiers, classify higher and flag for human review.
"""


# --- Index loading ----------------------------------------------------------

def load_index() -> VectorStoreIndex:
    """Open the Chroma collection."""
    db = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = db.get_collection(COLLECTION_NAME)
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def retrieve_chunks(index: VectorStoreIndex, description: str, top_k: int = TOP_K):
    """Return the top-k most relevant chunks for a system description."""
    retrieval_query = (
        f"{description} prohibited practices high-risk AI systems Annex III "
        f"transparency obligations Article 5 Article 50"
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(retrieval_query)


def _passage_label(node) -> str:
    """
    Build the passage header from chunk metadata. v3 indexes carry
    'provision' and 'provision_start_page'; fall back gracefully on
    older metadata so the module still works against a v2 index.
    """
    md = node.metadata
    provision = md.get("provision")
    start_page = md.get("provision_start_page")
    page = md.get("page", "?")
    if provision and start_page:
        if provision == "Recitals":
            return f"{provision}, page {page}"
        return f"{provision}, begins page {start_page} (this excerpt: page {page})"
    # Fallback for pre-v3 metadata.
    art = md.get("article_number", 0)
    if art:
        return f"Article {art}, page {page}"
    return f"Recitals, page {page}"


def build_user_prompt(description: str, retrieved_nodes) -> str:
    """Format description + retrieved chunks into the user message."""
    chunks_text = []
    for node in retrieved_nodes:
        label = _passage_label(node)
        text = node.text.replace("\n", " ").strip()
        chunks_text.append(f"[{label}]\n{text}")

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
        model=os.getenv("GROQ_MODEL", GROQ_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content

    return _parse_response(raw)


def _parse_response(raw: str) -> Classification:
    """Parse and validate the model's JSON response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
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
        return Classification(
            tier="minimal-risk",
            confidence="low",
            reasoning=f"Model returned unrecognised tier '{tier}'. Falling back to minimal-risk and flagging for review.",
            citations=[],
            needs_human_review=True,
            raw_response=raw,
        )

    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    reasoning = data.get("reasoning", "")
    model_flag = bool(data.get("needs_human_review", True))

    # Review-flag floor (v4). The v3 evaluation showed the model rates almost
    # every case "high" confidence, including boundary cases it gets wrong, so a
    # confidence-only floor rarely fired. v4 adds STRUCTURAL signals computed
    # from the reasoning text: if the model's own explanation mentions an
    # exception, a tier boundary, a borderline judgement, or a dependence on
    # interpretation, the case warrants human review regardless of the
    # confidence label. These are deterministic and do not rely on the model
    # choosing to set the flag itself.
    review_signals = (
        "borderline", "boundary", "near the", "close call", "could be",
        "depends on", "exception", "arguabl", "ambiguous", "unclear",
        "however", "on the other hand", "either", "contested", "provisional",
        "may also", "might also", "not clear", "uncertain",
    )
    text = reasoning.lower()
    structural_flag = any(sig in text for sig in review_signals)

    needs_review = model_flag or (confidence != "high") or structural_flag

    return Classification(
        tier=tier,
        confidence=confidence,
        reasoning=reasoning,
        citations=data.get("citations", []),
        needs_human_review=needs_review,
        raw_response=raw,
    )


# --- CLI entry point --------------------------------------------------------

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
