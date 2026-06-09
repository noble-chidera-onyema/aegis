"""
Aegis evaluation harness (Week 8).

Runs the classifier over a hand-labelled set of system descriptions and
reports measured accuracy. The labelled set (tests/eval_set.json) is the
author's ground truth, judged against the EU AI Act, not model-generated.

Because the free Groq tier meters tokens per day, the full 40-case run does
not fit in one window. This harness supports running in BATCHES: each run
scores a range of cases and saves them into a single combined store keyed by
case id (tests/eval_results/eval_combined.json). Running batches at different
times is statistically identical to running all at once; the cases are
independent. The store records, per case, when it ran and on which model, so
the result is auditable.

Measures (over whatever cases are present in the combined store):
  1. Tier accuracy: overall and per-tier (confusion matrix).
  2. Citation accuracy: Article-level, and Article+page (two numbers).
  3. Review-flag firing rate on boundary cases (the Week 4 defect).

Usage, from the project root:
    python -m tests.run_eval --range 1 15
    python -m tests.run_eval --range 16 30
    python -m tests.run_eval --range 31 40
    python -m tests.run_eval --range 1 40 --model llama-3.1-8b-instant
    python -m tests.run_eval --report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = PROJECT_ROOT / "tests" / "eval_set.json"
RESULTS_DIR = PROJECT_ROOT / "tests" / "eval_results"
COMBINED_PATH = RESULTS_DIR / "eval_combined.json"

TIERS = ("prohibited", "high-risk", "limited-risk", "minimal-risk")


def extract_article_keys(citation: str) -> set:
    keys = set()
    text = citation.lower()
    for m in re.finditer(r"article\s+(\d+)", text):
        keys.add(f"article {m.group(1)}")
    for m in re.finditer(r"annex\s+([ivx]+)", text):
        keys.add(f"annex {m.group(1)}")
    return keys


def extract_pages(citation: str) -> set:
    return {int(m.group(1)) for m in re.finditer(r"page\s+(\d+)", citation.lower())}


def citation_article_match(citations, expected_article) -> bool:
    want = extract_article_keys(expected_article)
    if not want:
        return False
    got = set()
    for c in citations:
        got |= extract_article_keys(c)
    return bool(want & got)


def citation_page_match(citations, expected_article) -> bool:
    want_keys = extract_article_keys(expected_article)
    want_pages = extract_pages(expected_article)
    if not want_keys or not want_pages:
        return False
    for c in citations:
        if (extract_article_keys(c) & want_keys) and (extract_pages(c) & want_pages):
            return True
    return False


def load_eval_set():
    if not EVAL_SET_PATH.exists():
        sys.exit(f"No eval set at {EVAL_SET_PATH}.")
    data = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    cases = [c for c in data.get("cases", []) if c.get("expected_tier", "").strip()]
    if not cases:
        sys.exit("Eval set has no labelled cases.")
    return cases


def load_combined():
    if COMBINED_PATH.exists():
        return json.loads(COMBINED_PATH.read_text(encoding="utf-8"))
    return {}


def save_combined(store):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def run_batch(lo, hi, model):
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Add it to your .env file.")
    if model:
        os.environ["GROQ_MODEL"] = model

    from groq import Groq
    from src.aegis.classify import classify_system
    from src.aegis.grounded_qa import load_index

    cases = load_eval_set()
    lo = max(1, lo)
    hi = min(len(cases), hi)
    batch = cases[lo - 1:hi]
    model_name = os.getenv("GROQ_MODEL", "default")
    print(f"Loaded {len(cases)} labelled cases. Running batch {lo}-{hi} ({len(batch)} cases).")
    print(f"Model: {model_name}")
    print("Loading index and client once...")

    client = Groq()
    index = load_index()
    store = load_combined()

    ran = 0
    failed = 0
    for offset, case in enumerate(batch):
        i = lo + offset
        cid = case.get("id", f"case{i}")
        expected = case["expected_tier"]
        print(f"  [{i}] {cid} (expect {expected})...", flush=True)

        result = None
        for attempt in range(2):
            try:
                result = classify_system(case["description"], client=client, index=index)
                break
            except Exception as exc:
                if attempt == 0:
                    time.sleep(2)
                    continue
                print(f"      classify failed: {exc}")

        if result is None:
            failed += 1
            continue

        exp_art = case.get("expected_article", "").strip()
        a_match = citation_article_match(result.citations, exp_art) if exp_art else None
        p_match = (citation_page_match(result.citations, exp_art)
                   if (exp_art and extract_pages(exp_art)) else None)

        store[cid] = {
            "id": cid,
            "expected_tier": expected,
            "got_tier": result.tier,
            "tier_correct": result.tier == expected,
            "confidence": result.confidence,
            "needs_human_review": result.needs_human_review,
            "difficulty": case.get("difficulty", ""),
            "expected_article": exp_art,
            "citation_article_match": a_match,
            "citation_page_match": p_match,
            "citations": result.citations,
            "ran_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
        }
        ran += 1
        save_combined(store)

    print(f"\nBatch done: {ran} classified, {failed} failed (left unstored for a later run).")
    print(f"Combined store now holds {len(store)} of {len(cases)} cases.")
    report(cases)


def report(cases=None):
    if cases is None:
        cases = load_eval_set()
    store = load_combined()
    total_cases = len(cases)
    done = [store[c["id"]] for c in cases if c.get("id") in store]
    n = len(done)

    print("\n" + "=" * 60)
    print(f"RESULTS  ({n} of {total_cases} cases run)")
    print("=" * 60)
    if n == 0:
        print("No cases run yet.")
        return

    tier_correct = sum(1 for r in done if r["tier_correct"])
    print(f"Tier accuracy: {tier_correct}/{n} = {tier_correct/n:.1%}")

    confusion = defaultdict(lambda: defaultdict(int))
    for r in done:
        confusion[r["expected_tier"]][r["got_tier"]] += 1

    print("\nPer-tier (expected down, got across):")
    print("  expected \\ got   " + "".join(f"{t[:9]:>11}" for t in TIERS))
    for exp in TIERS:
        line = f"  {exp:14}" + "".join(f"{confusion[exp][g]:>11}" for g in TIERS)
        tot = sum(confusion[exp].values())
        if tot:
            line += f"   ({confusion[exp][exp]}/{tot})"
        print(line)

    art = [r for r in done if r["citation_article_match"] is not None]
    if art:
        hits = sum(1 for r in art if r["citation_article_match"])
        print(f"\nCitation accuracy, Article-level: {hits}/{len(art)} = {hits/len(art):.1%}")
    page = [r for r in done if r["citation_page_match"] is not None]
    if page:
        hits = sum(1 for r in page if r["citation_page_match"])
        print(f"Citation accuracy, Article+page:  {hits}/{len(page)} = {hits/len(page):.1%}")

    boundary = [r for r in done if r["difficulty"] == "boundary"]
    if boundary:
        fired = sum(1 for r in boundary if r["needs_human_review"])
        print(f"\nReview-flag on boundary cases: {fired}/{len(boundary)} = {fired/len(boundary):.1%} fired")
        print("  (low here = the Week 4 underfire defect, measured)")

    if n < total_cases:
        missing = [c["id"] for c in cases if c.get("id") not in store]
        print(f"\nNOT YET RUN ({len(missing)}): {', '.join(missing)}")
        print("Run the remaining cases with --range once tokens free up.")
    else:
        print("\nAll cases run. This is the complete result.")


def main():
    ap = argparse.ArgumentParser(description="Aegis classifier evaluation harness.")
    ap.add_argument("--range", nargs=2, type=int, metavar=("LO", "HI"),
                    help="Run cases LO to HI (1-indexed, inclusive).")
    ap.add_argument("--model", default=None, help="Override GROQ_MODEL for this run.")
    ap.add_argument("--report", action="store_true",
                    help="Print the combined report without running anything.")
    args = ap.parse_args()

    if args.report:
        report()
    elif args.range:
        run_batch(args.range[0], args.range[1], args.model)
    else:
        run_batch(1, 9999, args.model)


if __name__ == "__main__":
    main()
