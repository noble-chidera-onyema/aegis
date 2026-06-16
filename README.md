# Aegis

An EU AI Act readiness tool for Irish small and medium businesses.

> Working name. "Aegis" is widely used in the AI-compliance space and will be
> changed to a distinctive, cleared name before public launch.

The Act's main obligations and enforcement powers take effect on 2 August 2026.
Most compliance tooling for the Act is built for large regulated enterprises
with legal teams and budgets to match. Aegis is aimed at the businesses those
tools ignore: the SME owner with one chatbot and a CV screener who needs to know,
in plain language, whether they are breaking the law.

Aegis takes a plain-language description of an AI system, classifies it against
the Act's four risk tiers (prohibited, high-risk, limited-risk, minimal-risk),
and returns the obligations that apply, each with a citation to the specific
Article or Annex of the Act. It also answers questions about the legislation by
retrieving the relevant passages and quoting the law rather than paraphrasing it.

Aegis is decision-support, not legal advice. Every output carries that
disclaimer and points the user toward qualified counsel. It is not affiliated
with the European Union or any Irish authority.

## What it does

- **Classify.** Describe an AI system in plain language; get a risk tier with a
  confidence level, grounded reasoning, citations to the governing Article or
  Annex, and a flag when the case warrants human review.
- **Obligations.** For the assigned tier, a report of the operative Articles,
  each with a source passage from the Act, the page it begins on, and a note
  explaining why it applies to this specific system.
- **Ask.** Grounded question-and-answer over the Act. Answers cite pages and
  decline when the retrieved passages do not support an answer.
- **Human in the loop.** A standing caution sits under every classification, and
  a stronger banner escalates when the system flags a case for review. The user
  can accept the tier or dispute it; a disputed classification marks the
  obligations report provisional. This follows the Act's own human-oversight
  Article, which names automation bias as the risk to guard against.

## How it works

The EU AI Act (144 pages) is parsed with pdfplumber and split on Article and
Annex boundaries, so each chunk knows which provision it belongs to and the page
that provision begins on. The chunks are embedded with
sentence-transformers/all-MiniLM-L6-v2 and stored in a local Chroma vector
store (232 chunks). For a query, a retrieval layer pulls the most relevant
passages; a Groq-hosted Llama 3.3 70B model takes the system description plus
those passages and returns a structured classification with citations.

Built in Python with Streamlit, LlamaIndex, Chroma, and Groq. Runs locally with
`streamlit run app.py`. Deployment target is Streamlit Community Cloud first,
then Fly.io Frankfurt for EU data residency before any public launch.

## Evaluation

The classifier is measured against a 40-case hand-labelled evaluation set, not
self-assessed. The method is the part that makes the number trustworthy:

- The 40 case descriptions were written to be adversarial: hidden tier
  boundaries, distractors, and systems that read like one tier but belong in
  another.
- The ground truth was derived by hand from the EU AI Act text, independent of
  the model under test. A model-written answer key would be the system grading
  itself, so the labels come from the Act, not the classifier.
- The harness reports tier accuracy (overall and per tier), citation accuracy at
  two levels (right Article/Annex, and right Article plus right page), and how
  often the human-review flag fires on boundary cases.

Three measured versions were run on the same set and the same independent ground
truth:

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Tier accuracy | 77.5% | 75.0% | 70.0% |
| Citation, right Article/Annex | 80.0% | 72.5% | 85.0% |
| Citation, Article + page | 36.8% | 42.1% | 44.7% |
| Review flag on boundary cases | 10% | 10% | 0% |

The honest reading: v2 fixed a page-metadata defect; v3's Annex-aware indexing
and citation convention gave the best citation accuracy of the three, but its
stricter tier prompt over-corrected and lowered tier accuracy. v3 is a tradeoff,
not a clean win, and it is documented as one. The evaluation also surfaced that
the review-flag underfire is a model-overconfidence problem, not a logic problem:
the model rates boundary cases "high" confidence, so a confidence-based flag
rarely fires.

At 40 cases the margin of error is about 13 points, so these are signals, not
precise benchmarks. Full write-ups: [docs/EVAL_RESULTS_v1.md](./docs/EVAL_RESULTS_v1.md)
and [docs/EVAL_RESULTS_v2_v3.md](./docs/EVAL_RESULTS_v2_v3.md).

## Known limitations

- Tier accuracy sits around the high 70s on a hard set; the prohibited tier has
  too few cases to give a stable rate and is treated as a spot check.
- Citation page precision is the weakest metric; the model names the right
  provision far more often than the exact page.
- The review-flag underfire (overconfidence) is not yet solved; the standing
  caution floor on every classification is the current mitigation.
- From v2 onward the 40-case set guided changes, so it is a development signal.
  A fresh held-out set is planned to confirm the chosen version generalises.

## Roadmap

- Revert the over-strict tier prompt while keeping v3's indexing and citation
  gains; re-measure.
- Address the review flag with structural signals rather than the model's
  self-reported confidence.
- Improve retrieval so the governing provision is reliably returned.
- Add a fresh held-out evaluation set.
- Deploy to EU-resident hosting before any public launch.

## Privacy

Session-only. Aegis stores nothing; inputs live in the browser session and are
gone when the tab closes. Inputs are sent to Groq to generate answers. Per
Groq's policy, inference requests are not retained by default and may be logged
briefly (up to 30 days) only for error or abuse investigation; data is encrypted
in transit and at rest. The in-app privacy notice says so plainly, and users are
warned not to enter personal data.

## Accessibility

Reviewed against WCAG 2.2 AA: text contrast computed for every colour pairing
and the one borderline value corrected; keyboard navigation reaches and operates
every control with a visible focus outline; form inputs carry accessible labels.

## Author

Noble Chidera Onyema, MSc Applied AI and User Experience, Abertay University.
onyemanoble1628@gmail.com
https://www.linkedin.com/in/noble-chidera-onyema-1a88b53ab/

## Licence

All Rights Reserved. See LICENSE.

## Build journey

Week-by-week history of the project, with screenshots:
[docs/BUILD_JOURNEY.md](./docs/BUILD_JOURNEY.md).
