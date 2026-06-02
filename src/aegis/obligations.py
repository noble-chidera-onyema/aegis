"""
Aegis obligations and gap report.

Given a Classification (from src/aegis/classify.py), produce a structured
report of the obligations that apply under the EU AI Act, with the Article
and page citation for each, and a plain-language note about why each
obligation applies to the user's specific system.

Three layers, each separately auditable:
1. The Article-to-tier mapping (hardcoded from the Act).
2. The source passage for each Article (retrieved from the Chroma index).
3. The plain-language "applies_because" explanation (LLM-generated, but
   constrained to the user's specific description and the retrieved passage).

If retrieval fails for an Article, the report still shows the hardcoded
title and description with a flag noting the source passage was not found.
If the LLM call fails, the report still shows the hardcoded obligation
without the per-system explanation. The user always gets something useful.

Week 6 Streamlit UI will import generate_report() directly.

Run from the project root for a CLI demonstration:
    python src/aegis/obligations.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import chromadb
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Import the Classification dataclass and tier type from the Week 4 module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify import Classification, Tier, load_index, classify_system  # noqa: E402


# --- Configuration ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ai_act_v1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
WRAP_WIDTH = 88

ComplianceLevel = Literal["required", "recommended", "voluntary", "cease"]


# --- The obligations rubric (hardcoded from the Act, auditable) ------------

# Each entry is one obligation. The `article` field is what the LLM cites and
# what retrieval looks up. The `description` field is hardcoded plain-language
# summary; it does NOT change between users. The retrieved source_passage and
# the LLM-generated applies_because layer on top of these at runtime.

@dataclass
class ObligationSpec:
    article: str
    title: str
    description: str
    checklist_question: str
    compliance_level: ComplianceLevel


PROHIBITED_OBLIGATIONS: list[ObligationSpec] = [
    ObligationSpec(
        article="Article 5",
        title="Cease all use of this system",
        description=(
            "Article 5 prohibits the practices this system implements. "
            "Continuing to place this system on the market or put it into "
            "service after 2 February 2025 risks the highest tier of "
            "administrative fines under Article 99 (up to EUR 35 million or "
            "7% of global annual turnover, whichever is higher)."
        ),
        checklist_question="Has all use of this system been ceased and have alternative non-AI processes been put in place?",
        compliance_level="cease",
    ),
]

HIGH_RISK_OBLIGATIONS: list[ObligationSpec] = [
    ObligationSpec(
        article="Article 9",
        title="Risk management system",
        description=(
            "Establish, document, and maintain a risk management system "
            "for the AI system across its entire lifecycle. The system must "
            "identify and analyse known and foreseeable risks, evaluate "
            "risks emerging from intended use and reasonably foreseeable "
            "misuse, and adopt risk management measures."
        ),
        checklist_question="Is there a documented risk management system covering the AI system's full lifecycle?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 10",
        title="Data and data governance",
        description=(
            "Training, validation, and testing datasets must be relevant, "
            "sufficiently representative, free of errors, and complete. "
            "Examine datasets for possible biases, address them through "
            "appropriate measures, and document data governance practices."
        ),
        checklist_question="Are training, validation, and testing data governance practices documented, including bias detection?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 11",
        title="Technical documentation",
        description=(
            "Maintain technical documentation that demonstrates compliance "
            "with the Act's requirements. The documentation must include "
            "general system description, design specifications, training "
            "methodologies, and monitoring procedures. Annex IV specifies "
            "the full contents."
        ),
        checklist_question="Is technical documentation complete and aligned with Annex IV contents?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 12",
        title="Record-keeping (logging)",
        description=(
            "Automatically log events relevant to identifying national-level "
            "risks and substantial modifications throughout the system's "
            "lifecycle. Logs must be retained for an appropriate period and "
            "must enable post-market monitoring."
        ),
        checklist_question="Does the system automatically log relevant events for post-market monitoring?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 13",
        title="Transparency and information to deployers",
        description=(
            "Design the system to be sufficiently transparent that deployers "
            "can interpret outputs and use them appropriately. Provide "
            "instructions for use that include the provider's identity, the "
            "system's characteristics and capabilities, performance limits, "
            "and known risks."
        ),
        checklist_question="Are instructions for use provided to deployers, covering all Article 13 requirements?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 14",
        title="Human oversight",
        description=(
            "Design and develop the system so that it can be effectively "
            "overseen by humans during the period in which it is in use. "
            "Oversight measures must enable individuals to understand the "
            "system's capacity and limits, monitor operation, and intervene "
            "or disregard the output."
        ),
        checklist_question="Are human oversight measures designed in and documented, including the ability to intervene or override?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 15",
        title="Accuracy, robustness, and cybersecurity",
        description=(
            "Achieve appropriate levels of accuracy, robustness, and "
            "cybersecurity, and perform consistently across the lifecycle. "
            "Declared accuracy levels and metrics must appear in the "
            "instructions for use. Resilience against unauthorised "
            "attempts to alter use, outputs, or performance is required."
        ),
        checklist_question="Are accuracy, robustness, and cybersecurity measures documented and tested?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 17",
        title="Quality management system",
        description=(
            "Establish a quality management system that ensures compliance "
            "with the Act. The system must include strategies for "
            "regulatory compliance, design control, verification and "
            "validation, and an accountability framework."
        ),
        checklist_question="Is a quality management system in place that meets Article 17 requirements?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 43",
        title="Conformity assessment",
        description=(
            "Before placing the system on the market or putting it into "
            "service, undergo the conformity assessment procedure relevant "
            "to the system's type. For most Annex III high-risk systems, "
            "this is internal control under Annex VI; some categories "
            "require involvement of a notified body."
        ),
        checklist_question="Has the appropriate conformity assessment been completed before market entry?",
        compliance_level="required",
    ),
    ObligationSpec(
        article="Article 49",
        title="Registration in the EU database",
        description=(
            "Before placing a high-risk AI system on the market, register "
            "the system in the EU database referred to in Article 71. The "
            "registration must include the information set out in Annex VIII."
        ),
        checklist_question="Is the system registered in the EU database with all Annex VIII information?",
        compliance_level="required",
    ),
]

LIMITED_RISK_OBLIGATIONS: list[ObligationSpec] = [
    ObligationSpec(
        article="Article 50",
        title="Transparency to natural persons",
        description=(
            "Design and develop the system so that natural persons are "
            "informed they are interacting with an AI system, unless this "
            "is obvious from the circumstances and context. For deepfakes "
            "and AI-generated content, disclose the artificial or "
            "manipulated nature of the content."
        ),
        checklist_question="Are users clearly informed they are interacting with an AI system?",
        compliance_level="required",
    ),
]

MINIMAL_RISK_OBLIGATIONS: list[ObligationSpec] = [
    ObligationSpec(
        article="Article 95",
        title="Voluntary codes of conduct",
        description=(
            "No mandatory obligations apply under the Act for this system. "
            "Article 95 encourages voluntary adoption of codes of conduct "
            "covering the Article 9-15 requirements (the high-risk regime), "
            "applied proportionately. Adoption signals trustworthiness to "
            "customers, investors, and regulators."
        ),
        checklist_question="Has the organisation considered adopting voluntary codes of conduct?",
        compliance_level="voluntary",
    ),
]

OBLIGATIONS_BY_TIER: dict[Tier, list[ObligationSpec]] = {
    "prohibited": PROHIBITED_OBLIGATIONS,
    "high-risk": HIGH_RISK_OBLIGATIONS,
    "limited-risk": LIMITED_RISK_OBLIGATIONS,
    "minimal-risk": MINIMAL_RISK_OBLIGATIONS,
}


# --- The runtime report shape ---------------------------------------------

@dataclass
class Obligation:
    article: str
    title: str
    description: str
    checklist_question: str
    compliance_level: ComplianceLevel
    source_passage: str  # the retrieved text from the Act, or "" if not found
    source_page: int | None  # page number from retrieval, or None
    applies_because: str  # LLM-generated per-system explanation, or "" if call failed


@dataclass
class ObligationsReport:
    tier: Tier
    system_description: str
    obligations: list[Obligation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --- Retrieval helper ------------------------------------------------------

def find_source_passage(index: VectorStoreIndex, article: str) -> tuple[str, int | None]:
    """
    Look up the Act's text for a specific Article using metadata filtering.

    Each chunk in the index is tagged with article_number metadata by
    build_index.py's Article-aware splitter. Citation lookup is therefore
    a direct metadata query, not a text search. This replaces the earlier
    heuristic approach (position + density scoring), which was unreliable
    because chunk boundaries sometimes separated Article headers from
    their bodies.

    Returns: (text, page) for the Article's first chunk, or ("", None)
    if the Article number cannot be parsed or no matching chunk exists.
    """
    db = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = db.get_collection(COLLECTION_NAME)

    # Parse the Article number out of the string ("Article 9" -> 9).
    try:
        article_number = int(article.replace("Article ", "").strip())
    except ValueError:
        return "", None

    # Metadata query. sub_chunk_index=0 is the first sub-chunk of the
    # Article, which always contains the header and the opening paragraphs.
    results = collection.get(
        where={
            "$and": [
                {"article_number": {"$eq": article_number}},
                {"sub_chunk_index": {"$eq": 0}},
            ]
        },
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        # Fallback: any chunk for this Article. Covers the edge case where
        # sub_chunk_index numbering got perturbed.
        results = collection.get(
            where={"article_number": {"$eq": article_number}},
            include=["documents", "metadatas"],
        )

    if not results["documents"]:
        return "", None

    doc_text = results["documents"][0]
    meta = results["metadatas"][0]
    page = meta.get("page")
    return doc_text.strip(), int(page) if page is not None else None


# --- LLM helper for the per-system explanation -----------------------------

EXPLANATION_SYSTEM_PROMPT = """You are a compliance research assistant. The user has been told their AI system is subject to a specific obligation from the EU AI Act. Your job is to write a short note (2 to 3 sentences) explaining why this particular obligation applies to their specific system description.

Rules:
1. Stay grounded in the user's description. Do not invent details about their system.
2. Reference the obligation's Article and its general purpose, not the specific text of the Act.
3. Be concrete. Tell the user what THIS obligation will look like for THEIR system.
4. If the obligation's relevance is obvious from the description, say so briefly.
5. Output plain text only. No JSON. No markdown. No preamble.

Example output: "Because your system ranks job candidates, the risk management process will need to cover known risks of bias in CV-screening algorithms, and document mitigation steps from the training data onward."
"""


def explain_obligation(client: Groq, spec: ObligationSpec, system_description: str) -> str:
    """Generate a 2-3 sentence per-system explanation for a single obligation."""
    user_prompt = (
        f"System description: {system_description}\n\n"
        f"Obligation: {spec.article} - {spec.title}\n"
        f"What the obligation requires: {spec.description}\n\n"
        f"Write the 2 to 3 sentence note."
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        # Defensive. The hardcoded description is still shown; the per-system note is just missing.
        return f"(Explanation unavailable: {exc})"


# --- The public function --------------------------------------------------

def generate_report(classification: Classification, system_description: str,
                    client: Groq | None = None,
                    index: VectorStoreIndex | None = None) -> ObligationsReport:
    """
    Generate an ObligationsReport for a classified AI system.

    Args:
        classification: result from classify_system().
        system_description: the original plain-language description.
        client: optional Groq client. Created from env if not supplied.
        index: optional VectorStoreIndex. Loaded from disk if not supplied.

    Returns:
        ObligationsReport with one Obligation per applicable Article.
    """
    if client is None:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        client = Groq(api_key=api_key)

    if index is None:
        index = load_index()

    specs = OBLIGATIONS_BY_TIER.get(classification.tier, [])
    report = ObligationsReport(
        tier=classification.tier,
        system_description=system_description,
    )

    if classification.needs_human_review:
        report.notes.append(
            "The underlying classification was flagged for human review. "
            "These obligations are advisory; verify the tier with qualified counsel."
        )

    if classification.confidence == "low":
        report.notes.append(
            "Classifier confidence is low. The obligations below assume the "
            "tier is correct; if it is not, a different obligation set applies."
        )

    for spec in specs:
        source_text, source_page = find_source_passage(index, spec.article)
        explanation = explain_obligation(client, spec, system_description)

        report.obligations.append(Obligation(
            article=spec.article,
            title=spec.title,
            description=spec.description,
            checklist_question=spec.checklist_question,
            compliance_level=spec.compliance_level,
            source_passage=source_text,
            source_page=source_page,
            applies_because=explanation,
        ))

    return report


# --- CLI demonstration ----------------------------------------------------

TEST_DESCRIPTIONS = [
    {
        "label": "1. Hiring CV screener (high-risk path)",
        "description": (
            "We are a 40-person company in Dublin. We use an AI tool that "
            "reads incoming CVs and ranks candidates by predicted fit with "
            "the role. The shortlist goes to a human recruiter who makes "
            "the final interview decisions."
        ),
    },
    {
        "label": "2. Customer service chatbot (limited-risk path)",
        "description": (
            "We run an Irish e-commerce site. We have an AI chatbot that "
            "answers customer questions about orders, returns, and product "
            "availability. The chatbot is clearly labelled as automated."
        ),
    },
]


def _print_report(report: ObligationsReport) -> None:
    print(f"\n  Tier:          {report.tier}")
    print(f"  Obligations:   {len(report.obligations)}\n")

    for note in report.notes:
        print(textwrap.fill(f"  Note: {note}", width=WRAP_WIDTH,
                            subsequent_indent="        "))
        print()

    for i, ob in enumerate(report.obligations, start=1):
        print(f"  [{i}] {ob.article}: {ob.title}")
        print(f"      Compliance level: {ob.compliance_level}")
        print(f"      Source page in Act: {ob.source_page if ob.source_page else 'not found'}")
        print()
        print(textwrap.fill(f"What this requires: {ob.description}",
                            width=WRAP_WIDTH,
                            initial_indent="      ", subsequent_indent="      "))
        print()
        print(textwrap.fill(f"Why this applies to your system: {ob.applies_because}",
                            width=WRAP_WIDTH,
                            initial_indent="      ", subsequent_indent="      "))
        print()
        print(textwrap.fill(f"Checklist question: {ob.checklist_question}",
                            width=WRAP_WIDTH,
                            initial_indent="      ", subsequent_indent="      "))
        print()


def main() -> int:
    print("Loading Chroma index and Groq client...")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.")
        return 1

    client = Groq(api_key=api_key)
    index = load_index()
    print("Ready.\n")

    for case in TEST_DESCRIPTIONS:
        print("=" * WRAP_WIDTH)
        print(case["label"])
        print("=" * WRAP_WIDTH)

        # First, classify (uses the Week 4 module).
        classification = classify_system(case["description"], client=client, index=index)
        print(f"\n  Classified as: {classification.tier} ({classification.confidence} confidence)")

        # Then, generate the obligations report.
        report = generate_report(classification, case["description"],
                                  client=client, index=index)
        _print_report(report)
        print()

    print("Obligations report is working end to end.")
    print("This is decision-support, not legal advice. Verify with qualified counsel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())