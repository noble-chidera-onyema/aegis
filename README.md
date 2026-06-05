# Aegis

An EU AI Act readiness tool for Irish small and medium businesses.

The Act's main obligations and enforcement powers take effect on 2 August 2026. Aegis is a free web app that takes a plain-language description of an AI system, classifies it against the Act's risk tiers (prohibited, high-risk, limited-risk, minimal-risk), and returns the obligations that apply, with citations to the actual articles. It also answers questions about the legislation using retrieval over the official text, so the answers quote the law rather than paraphrasing it.

The tool is decision-support, not legal advice. Every output carries that disclaimer and points the user toward qualified counsel.

## What works today

Behind the scenes: a working retrieval pipeline, a grounded question-and-answer layer, and a risk classifier. The EU AI Act (144 pages) is split into 296 paragraph-sized chunks embedded with sentence-transformers/all-MiniLM-L6-v2 and stored in a local Chroma vector database. The script in `src/aegis/grounded_qa.py` takes a question, retrieves the top five passages, sends them to Groq with Llama 3.3 70B, and prints an answer with `[page N]` citations. The module at `src/aegis/classify.py` takes a plain-language description of an AI system and returns a structured classification into one of four tiers (prohibited, high-risk, limited-risk, minimal-risk) with confidence, reasoning, citations, and a `needs_human_review` flag. Hand-checked on three sample questions and four sample classifications, all matching their expected tiers.

In the UI: all four screens work end to end. Inventory takes a plain-language system description, Classification shows the tier with grounded reasoning and cited Articles, Obligations shows the full report for that tier, and Ask answers questions about the legislation with page citations. Built in Week 6.

Known limits at this stage. pypdf introduces letter-spacing artefacts on the CELEX-format Act ("high-r isk", "Ar ticle"). Top-5 retrieval similarity scores sit between 0.35 and 0.50. The classifier returns inconsistent citation formatting (real page numbers when classifying INTO a tier, missing pages when ruling a tier OUT) and `needs_human_review` underfires on borderline cases like AI-assisted CV ranking. All flagged in `tests/test_classifications.md`. The Week 8 evaluation harness will measure these against a hand-labelled set of 30 to 50 cases and the upgrade path (better PDF extraction, larger embedding model, hybrid search with BM25, tighter prompt template) gets decided then on measured numbers.

## How it will work

The Act, Annex III, the Irish General Scheme of the AI Regulation Bill, and the GPAI Code of Practice are chunked and embedded into a local Chroma vector store. A retrieval layer pulls the most relevant clauses for each query. A Groq-hosted Llama model takes the user's system description plus the retrieved clauses and returns a structured classification with citations. A separate evaluation harness scores the classifier against a hand-labelled set so the accuracy claim in the README is a real number, not a marketing one.

Built in Python with Streamlit, LlamaIndex, Chroma, and Groq. Will deploy on Streamlit Community Cloud first, then move to Fly.io Frankfurt for EU data residency before the public launch.

## Privacy

Session-only. Aegis stores nothing; inputs live in the browser session and are gone when the tab closes. Inputs are sent to Groq to generate answers. Per Groq's policy, inference requests are not retained by default and may be logged briefly (up to 30 days) only for error or abuse investigation; data is encrypted in transit and at rest. The in-app privacy notice and a dedicated privacy section on the Inventory screen say so plainly. Users are warned not to enter personal data.

## Author

Noble Chidera Onyema, MSc Applied AI and User Experience, Abertay University.
onyemanoble1628@gmail.com
https://www.linkedin.com/in/noble-chidera-onyema-1a88b53ab/

## Licence

All Rights Reserved. See LICENSE.

## Build journey

Week-by-week history of the project, with screenshots: see [docs/BUILD_JOURNEY.md](./docs/BUILD_JOURNEY.md).
## Week 5: obligations report (live)

For any classified system, Aegis now returns an obligations report:
- the operative Articles that apply to that tier
- a verifiable source passage from the Act for each Article, with the
  page number where the Article begins in the source PDF
- a per-system note explaining why the Article applies to this specific
  system

Run with `python src/aegis/obligations.py`. Citation accuracy: 11 of 11
page references audited against `data/ai_act.pdf`. See `tests/test_obligations.md`.

## Week 6: Streamlit interface (live)

All four screens work end to end in the browser:

- Inventory: describe an AI system in plain language. A privacy warning sits next to the input; nothing is stored.
- Classification: returns the risk tier with confidence, grounded reasoning, and cited Articles.
- Obligations: the full obligations report for the tier. Each Article has a page citation, a per-system note explaining why it applies, and a checklist question.
- Ask: grounded Q&A over the Act. Questions that name an Article retrieve that Article by metadata filter; classification questions lead with Article 6; other questions use semantic search with front-matter excluded. Answers cite pages and decline when the retrieved passages do not support an answer.

A classified system carries across all four tabs through shared session state. Each screen runs inside an error boundary that preserves the user's inputs and shows a plain message on failure rather than crashing. The Groq model is environment-configurable via `GROQ_MODEL`, defaulting to llama-3.3-70b-versatile.

Run with `streamlit run app.py` from the project root.

Known limits, scheduled for the Week 8 evaluation harness: conceptual questions that name no Article rely on semantic retrieval, whose quality across a wide question set has not been measured systematically yet; top-k is fixed at 5; the Article-header detector found 114 headers against 113 operative Articles. All citation claims that appear in screenshots are verified against `data/ai_act.pdf` before publishing.

## Week 7: human-in-the-loop, privacy, accessibility (live)

The tool now keeps a person in the loop on every classification. A standing caution sits under each result, and a stronger banner escalates when the classifier flags a case for review. The user can accept the tier or flag disagreement, and that decision carries to the Obligations report, which marks itself provisional when the classification is disputed. This is aligned with the EU AI Act's own human-oversight Article, which names automation bias as the risk to guard against.

Each screen now leads with a single scannable line and a collapsible "How this works" panel, so the detail is there without burying the task. A dedicated privacy section on the Inventory screen states what Aegis and Groq do with the user's input, checked against Groq's current policy.

Accessibility was reviewed against WCAG 2.2 AA: text contrast was computed for every colour pairing and the one borderline value was corrected, keyboard navigation reaches and operates every control with a visible focus outline, and form inputs carry accessible labels.

Run with `streamlit run app.py` from the project root.