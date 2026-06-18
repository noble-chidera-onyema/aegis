# Evaluation results, v1 to v4

Last run: 18 June 2026
Model tested throughout: `llama-3.3-70b-versatile` (the model the app serves)
Eval set: the same 40 hand-labelled adversarial cases (`tests/eval_set.json`)
Ground truth: derived by hand from the EU AI Act, independent of the classifier
Raw results: `eval_combined_v1.json` through `eval_combined_v4.json`

## Timeline

Each version's 40 cases were run in batches across several token windows,
because the free Groq tier (100k tokens per day) does not fit a full 40-case
run in one day. The calendar spread below reflects that constraint, not the
work itself.

- v1 complete: 9 June 2026
- v2 built and run: 9 June 2026 (same day, after the page-metadata fix)
- v3 complete: 14 June 2026
- v4 complete: 18 June 2026

Four measured iterations over ten days, each one changed in response to the
previous measurement.

## The four versions

| Metric | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| Tier accuracy | 77.5% | 75.0% | 70.0% | 70.0% |
| Citation, right Article/Annex | 80.0% | 72.5% | 85.0% | 72.5% |
| Citation, Article + page | 36.8% | 42.1% | 44.7% | 47.4% |
| Limited-risk recall | over-assigned | n/a | 2/7 | 5/7 |
| Review flag on boundary cases | 10% | 10% | 0% | 10% |

There is no version that is best on every metric. v1 has the best tier
accuracy. v3 has the best Article-level citation. v4 has the best page-level
citation and the most balanced limited-risk recall. Each change bought a gain
on one metric and gave something back on another.

## What each version changed and what it did

### v1, baseline
The classifier as first built. 77.5% tier accuracy on a deliberately hard set.

### v2, page-metadata fix
The ingestion code hardcoded recital chunks to page 1 and gave every sub-chunk
of a long Article the Article's start page. Fixed to compute each chunk's real
page. Article+page citation rose 36.8 to 42.1. Tier accuracy unchanged within
noise. The fix was real but modest, because most of the page mismatch was a
labelling-convention difference on Annex III, not the page-1 bug.

### v3, Annex-aware indexing, citation convention, tier procedure
Annexes became first-class chunks with their own start pages (Annex III at page
127), the prompt was told to cite the page a provision begins on, and a strict
tier decision procedure was added. Article-level citation reached its best
(85.0%) and tiering fell to its worst (70.0%): the strict procedure told the
model not to use limited-risk as a compromise, and it over-corrected, dropping
limited-risk recall to 2 of 7.

### v4, rebalanced limited-risk, structural review flag
The limited-risk step was rewritten to state the Article 50 triggers as
positive signals rather than discouraging the tier. The review flag was changed
to fire on structural signals in the model's reasoning (mentions of exceptions,
boundaries, hedging) rather than only on the model's self-reported confidence.

Result: limited-risk recall recovered to 5 of 7, the main goal. Article+page
citation reached its best (47.4%). But tier accuracy stayed at 70.0%, because
the rebalanced prompt now slightly over-reaches into limited-risk from
minimal-risk (3 minimal cases went to limited). Article-level citation dropped
back to 72.5%; the prompt change shifted which provisions the model emphasised.
The review flag recovered to 10% but still underfires.

## v4 confusion matrix (tier)

Rows are the labelled tier, columns are what the classifier returned.

| labelled \ got | prohibited | high-risk | limited-risk | minimal-risk | recall |
|---|---|---|---|---|---|
| prohibited   | 0 | 2  | 0 | 0  | 0/2 (spot check) |
| high-risk    | 1 | 13 | 2 | 1  | 13/17 = 76% |
| limited-risk | 0 | 2  | 5 | 0  | 5/7 = 71% |
| minimal-risk | 0 | 1  | 3 | 10 | 10/14 = 71% |

## Why iteration stops at v4

Four versions show a clear pattern: each prompt-and-index change fixes one
failure and introduces another, and the overall tier accuracy sits in a band
around the low-to-mid 70s. This is diminishing returns on prompt and index
tuning. Continuing to tune would most likely move the bottleneck again rather
than lift the ceiling.

There is also an integrity reason to stop. The same 40 cases have now guided
four rounds of change. Each further version tuned against them makes the
40-case number a weaker measure of real performance, because the prompt starts
fitting the test. The honest next step is not a v5 prompt tweak; it is a fresh
held-out set to check the chosen version generalises, and changes of a different
kind (retrieval quality, a stronger model, calibration of the confidence signal)
rather than more of the same.

## The review-flag finding (holds across all four versions)

The flag underfires because the model is overconfident, not because the flag
logic is wrong. Across versions the model rated almost every case "high"
confidence, including boundary cases it got wrong. v4's structural-signal
approach helped slightly (0% to 10%) but did not solve it. This is a model
calibration problem and is recorded as future work.

## Chosen version for deployment

v4 is the deployment candidate. Reasoning: for a decision-support tool, the
user reads the cited passage to confirm the answer, so citation quality matters
alongside the tier, and v4 has the best page-level citation and a balanced
limited-risk recall. v1's higher raw tier accuracy is noted, and the gap is
within the set's margin of error (about 13 points at 40 cases), so the two are
not clearly distinguishable on tiering alone. The decision favours the version
with the more useful citations and the more balanced tier behaviour.

## What comes next (not more tuning)

1. A fresh held-out set of new cases, labelled the same independent way, to
   confirm v4 generalises and was not fitted to these 40.
2. Retrieval quality: the governing provision is not always returned in the top
   results, which weakens citations more than the prompt does.
3. Confidence calibration, so the review flag has an honest signal to act on.
4. Deployment to EU-resident hosting.
