# Evaluation results, v1 to v3

Last run: 14 June 2026
Model tested throughout: `llama-3.3-70b-versatile` (the model the app serves)
Eval set: the same 40 hand-labelled adversarial cases (`tests/eval_set.json`)
Ground truth: derived by hand from the EU AI Act, independent of the classifier
Raw results: `eval_combined_v1.json`, `eval_combined_v2.json`, `eval_combined.json` (v3)

## The three versions at a glance

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| Tier accuracy | 77.5% (31/40) | 75.0% (30/40) | 70.0% (28/40) |
| Citation, Article level | 80.0% | 72.5% | 85.0% |
| Citation, Article + page | 36.8% | 42.1% | 44.7% |
| Review flag on boundary cases | 10% (1/10) | 10% | 0% (0/10) |

The honest headline: v3 has the best citations and the worst tiering. It is a
tradeoff, not a clean win. Each change below is recorded with what it was meant
to do and what it actually did.

## What changed in each version

### v1, baseline
The classifier as first built. 77.5% tier accuracy on a set written to be hard.
This is the reference point.

### v2, page-metadata fix
The v1 evaluation showed citations sometimes reported "page 1" for text that
was plainly not on page 1. Tracing it: the ingestion code hardcoded every
recital chunk to page 1, and every sub-chunk of a long Article inherited the
Article's start page rather than its own.

Fix: compute the real page of every chunk from a character-to-page map.
Verified: recital chunks that reported page 1 now report their true pages (24,
35, 42).

Result: Article+page citation moved from 36.8% to 42.1%. Tier accuracy was
unchanged within noise (77.5 to 75.0). The page fix was real but modest,
because the bulk of the page mismatch was never the page-1 bug. It was a
labelling-convention difference on Annex III (see below), which this fix did
not touch.

### v3, Annex-aware indexing, citation convention, tier procedure, review floor
Four changes, driven by the v1 and v2 results.

1. Annex-aware indexing. v1 and v2 only detected "Article N" headers, so all
   Annex text was glued onto the last Article's chunks with no Annex metadata.
   Annex III, the most-cited provision for high-risk cases, was structurally
   invisible. v3 detects all 13 annexes as their own chunks with their own
   start pages. Annex III is correctly tagged as beginning on page 127.

2. Citation convention. Passages are now presented to the model with the page
   where their provision begins, and the model is told to cite that page. This
   matches the convention the ground truth uses (the page a reader flips to).

3. Tier decision procedure. An explicit four-step procedure in the prompt,
   aimed at the v1 error pattern of over-assigning limited-risk, with the
   Article 5 significant-harm gate and the main exceptions spelled out.

4. Review-flag floor. The flag now fires in code whenever model confidence is
   below "high", not only when the model sets it.

## What v3 actually did, measured

Citations improved, as intended. Article-level reached 85.0% (best of the
three): the Annex-aware indexing helps the model name the right provision.
Article+page reached 44.7% (best of the three): the start-page convention
moved it up about 8 points over v1.

Tiering regressed. Tier accuracy fell to 70.0%. The cause is visible in the
confusion matrix: limited-risk dropped to 2 of 7. The decision-procedure rule
("require a real Article 50 trigger, do not use limited-risk as a compromise")
over-corrected. v1 over-assigned limited-risk; v3 under-assigns it, sending
limited-risk cases to minimal-risk and high-risk instead. The same tier, the
opposite failure, and on balance worse.

Prohibited stayed at 0 of 2 across all three versions (both cases classified as
high-risk). The significant-harm gate language did not change this.

The review flag fell to 0 of 10 on boundary cases. This is the most useful
finding of the three versions, explained next.

## The review-flag finding (holds across all three versions)

The flag underfires not because the flag logic is wrong, but because the model
is overconfident. A dump of the stored confidence values showed the model
rating 16 of 17 cases "high" confidence, including every boundary case, the
cases it then gets wrong. The v3 floor fires when confidence is below "high",
so it almost never triggers, because the model rarely reports less than high
confidence.

So the review-flag problem is a confidence-calibration problem, not a flag-logic
problem. No amount of flag wiring fixes a model that will not report doubt. This
is consistent across v1, v2, and v3.

## v3 confusion matrix (tier)

Rows are the labelled tier, columns are what the classifier returned.

| labelled \ got | prohibited | high-risk | limited-risk | minimal-risk | recall |
|---|---|---|---|---|---|
| prohibited   | 0 | 2  | 0 | 0  | 0/2 (spot check) |
| high-risk    | 1 | 13 | 1 | 2  | 13/17 = 76% |
| limited-risk | 0 | 2  | 2 | 3  | 2/7 = 29% |
| minimal-risk | 0 | 1  | 0 | 13 | 13/14 = 93% |

## Honest reading

v3 is not strictly better than v1. It buys better citations at the cost of
worse tiering. If the product's priority is correct risk tiers, v1's prompt was
better and v3's tier rule should be rolled back. If the priority is trustworthy
citations, v3's indexing and citation convention are the keepers. The right
v4 is most likely v3's indexing and citation work, with v1's lighter tier
prompt, not v3's aggressive one.

## Limitation introduced by this process

From v2 onward, the 40 cases were used to guide changes, so the set is now a
development signal, not a held-out test. The version-to-version numbers are
comparable to each other, but they may flatter the system slightly against truly
unseen inputs. Before submission, a small fresh holdout (8 to 10 new cases,
labelled the same independent way) should be run on the chosen version to check
it generalises and was not tuned to these 40.

## What v4 should do, in order

1. Revert the limited-risk tier rule to v1's lighter prompt; keep v3's indexing
   and citation convention. Re-run to confirm citations stay high while tiering
   returns toward 77%.
2. Address the review flag through structural signals (for example, firing when
   the reasoning itself names an exception or a tier boundary) rather than the
   model's self-reported confidence.
3. Improve retrieval so the governing provision is actually returned. v3's
   metadata exposed that a query like "rank job candidates" does not return
   Annex III in the top eight chunks; the model reaches the right tier from
   neighbouring text, which weakens citations.
4. Add the fresh holdout described above.
