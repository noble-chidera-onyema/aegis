# Aegis

An EU AI Act readiness tool for Irish small and medium businesses.

The Act's main obligations and enforcement powers take effect on 2 August 2026. Aegis is a free web app that takes a plain-language description of an AI system, classifies it against the Act's risk tiers (prohibited, high-risk, limited-risk, minimal-risk), and returns the obligations that apply, with citations to the actual articles. It also answers questions about the legislation using retrieval over the official text, so the answers quote the law rather than paraphrasing it.

The tool is decision-support, not legal advice. Every output carries that disclaimer and points the user toward qualified counsel.

## What works today

Behind the scenes: a working retrieval pipeline, a grounded question-and-answer layer, and a risk classifier. The EU AI Act (144 pages) is split into 296 paragraph-sized chunks embedded with sentence-transformers/all-MiniLM-L6-v2 and stored in a local Chroma vector database. The script in `src/aegis/grounded_qa.py` takes a question, retrieves the top five passages, sends them to Groq with Llama 3.3 70B, and prints an answer with `[page N]` citations. The module at `src/aegis/classify.py` takes a plain-language description of an AI system and returns a structured classification into one of four tiers (prohibited, high-risk, limited-risk, minimal-risk) with confidence, reasoning, citations, and a `needs_human_review` flag. Hand-checked on three sample questions and four sample classifications, all matching their expected tiers.

In the UI: still a Streamlit "Hello World" page. The user-facing features come next. Obligations report is Week 5. Streamlit interface wiring is Week 6.

Known limits at this stage. pypdf introduces letter-spacing artefacts on the CELEX-format Act ("high-r isk", "Ar ticle"). Top-5 retrieval similarity scores sit between 0.35 and 0.50. The classifier returns inconsistent citation formatting (real page numbers when classifying INTO a tier, missing pages when ruling a tier OUT) and `needs_human_review` underfires on borderline cases like AI-assisted CV ranking. All flagged in `tests/test_classifications.md`. The Week 8 evaluation harness will measure these against a hand-labelled set of 30 to 50 cases and the upgrade path (better PDF extraction, larger embedding model, hybrid search with BM25, tighter prompt template) gets decided then on measured numbers.

## How it will work

The Act, Annex III, the Irish General Scheme of the AI Regulation Bill, and the GPAI Code of Practice are chunked and embedded into a local Chroma vector store. A retrieval layer pulls the most relevant clauses for each query. A Groq-hosted Llama model takes the user's system description plus the retrieved clauses and returns a structured classification with citations. A separate evaluation harness scores the classifier against a hand-labelled set so the accuracy claim in the README is a real number, not a marketing one.

Built in Python with Streamlit, LlamaIndex, Chroma, and Groq. Will deploy on Streamlit Community Cloud first, then move to Fly.io Frankfurt for EU data residency before the public launch.

## Privacy

Session-only. Nothing is stored. Inputs are sent to Groq for inference and may be retained by them for up to 30 days per their terms. The in-app privacy notice says so plainly. Users are warned not to enter personal data.

## Author

Noble Chidera Onyema, MSc Applied AI and User Experience, Abertay University.
onyemanoble1628@gmail.com
https://www.linkedin.com/in/noble-chidera-onyema-1a88b53ab/

## Licence

All Rights Reserved. See LICENSE.

## Build journey

Week-by-week history of the project, with screenshots: see [docs/BUILD_JOURNEY.md](./docs/BUILD_JOURNEY.md).