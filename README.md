# Aegis

An EU AI Act readiness tool for Irish small and medium businesses.

The Act's main obligations and enforcement powers take effect on 2 August 2026. Aegis is a free web app that takes a plain-language description of an AI system, classifies it against the Act's risk tiers (prohibited, high-risk, limited-risk, minimal-risk), and returns the obligations that apply, with citations to the actual articles. It also answers questions about the legislation using retrieval over the official text, so the answers quote the law rather than paraphrasing it.

The tool is decision-support, not legal advice. Every output carries that disclaimer and points the user toward qualified counsel.

## What works today

Behind the scenes: a working retrieval pipeline. The EU AI Act (144 pages) is split into 296 paragraph-sized chunks, each embedded with sentence-transformers/all-MiniLM-L6-v2. The chunks plus their embeddings live in a local Chroma vector database. A smoke test in `src/aegis/test_retrieval.py` confirms sample questions return the relevant passages of the Act with page numbers attached.

In the UI: still a Streamlit "Hello World" page. The user-facing features come next. Grounded Q&A is Week 3. Risk classification is Week 4. Obligations report is Week 5.

Known limits at this stage. pypdf introduces letter-spacing artefacts on the CELEX-format Act ("high-r isk", "Ar ticle"). Top-3 similarity scores currently sit between 0.36 and 0.53. The Week 8 evaluation harness will measure these properly against a hand-labelled set, and the upgrade path (better PDF extraction, larger embedding model, or hybrid search with BM25) gets decided then based on measured numbers.

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