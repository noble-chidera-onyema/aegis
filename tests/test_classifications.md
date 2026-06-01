# Aegis risk classifier test cases

Hand-crafted descriptions covering each of the four EU AI Act risk tiers, used to spot-check whether `src/aegis/classify.py` returns the correct classification with grounded reasoning. Not the formal evaluation set. The Week 8 evaluation harness will use 30 to 50 hand-labelled descriptions and measure classification accuracy and citation accuracy properly.

Verified manually on 1 June 2026 by running the script and reading the output.

## Case 1: Hiring CV screener

Description: a 40-person Dublin company uses an AI tool that reads incoming CVs and ranks candidates by predicted fit, with a human recruiter making the final interview decisions.

Expected tier: high-risk.

Reasoning Annex III(4)(a) covers AI systems used in recruitment, in particular for screening or filtering applications and evaluating candidates. The human-in-the-loop on the final decision does not change the classification, because the AI still ranks and shortlists.

Observed 1 June 2026: tier high-risk, confidence high, reasoning cited Article 6 and Annex III with real page numbers (page 53 and page 130). Match.

Calibration note: the model returned needs_human_review false. A real compliance officer would still flag this for review because the boundary between "AI-assisted ranking" and "AI-driven decision-making" is contested in practice. The classifier is more confident than the underlying legal position warrants. This is flagged for Week 8 evaluation.

## Case 2: Customer service chatbot

Description: an Irish e-commerce site runs an AI chatbot that answers customer questions about orders, returns, and product availability. The chatbot is clearly labelled as automated.

Expected tier: limited-risk.

Reasoning Article 50(1) places transparency obligations on AI systems that interact with natural persons. The disclosure "this is automated" is itself the compliance act. No high-risk Annex III category applies. No Article 5 prohibition applies.

Observed 1 June 2026: tier limited-risk, confidence high, reasoning cited Article 50 page 97 and Article 5 page 51. Match.

## Case 3: Social scoring system

Description: an AI system that scores citizens based on social media activity, payment history, and public behaviour records, and uses the score to decide access to public services.

Expected tier: prohibited.

Reasoning Article 5(1)(c) prohibits social scoring by public authorities or on their behalf that leads to detrimental or unfavourable treatment of natural persons in social contexts unrelated to those in which the data was originally generated, or that is unjustified or disproportionate to the social behaviour.

Observed 1 June 2026: tier prohibited, confidence high, reasoning correctly identified the practice as social scoring. Match.

Citation quality note: the model cited "Article 5, page not specified" and "Passage 3, page 9". The Article 5 citation is missing the page number even though the system prompt asks for it. The "Passage 3" reference is an internal retrieval label, not an article reference an end user can verify. Citation formatting is inconsistent when the relevant Article appears only partially in the retrieved chunks. Flagged for Week 8 evaluation; the fix is likely a tighter prompt template requiring strict citation format.

## Case 4: Internal spam filter

Description: a 12-person law firm uses an AI spam filter on the company email server.

Expected tier: minimal-risk.

Reasoning No Annex III category applies. No Article 5 prohibition applies. The system does not interact with natural persons in a way that triggers Article 50. Most AI systems fall here, and so should this one.

Observed 1 June 2026: tier minimal-risk, confidence high, reasoning correctly ruled out each higher tier in turn. Match.

Citation quality note: all three citations returned "page not specified". When the model rules a tier OUT, it cites the Article generically without finding a specific passage. This is the same defect pattern as Case 3 and confirms it is a consistent prompt-template issue, not a one-off. Flagged for Week 8 evaluation.

## Cases queued for the Week 8 evaluation harness

The formal evaluation set in Week 8 will include adversarial cases. Examples planned:

- A real-time biometric identification system at a stadium (testing the Article 5(1)(h) boundary).
- A medical device safety component (testing the Annex I path to high-risk that is different from the Annex III path).
- An AI tutor that adapts to learning styles (testing the Annex III(3)(a) education boundary against minimal-risk).
- A deepfake video generator (testing Article 50 transparency obligations specifically).
- A description that contains no AI at all (testing that the classifier correctly refuses to classify and asks for clarification).
- A description that mixes a prohibited use with a benign use (testing whether the model returns the higher-stakes tier).

## Known limitations

- needs_human_review is currently returning false too often. The model is overconfident for borderline cases like the CV screener. Week 8 will retune the system prompt with explicit examples of when to flag for review.
- Citation formatting is inconsistent when the model rules a tier out vs rules it in. Pages are present when classifying INTO a tier, missing when classifying OUT of a tier. Same Week 8 fix.
- pypdf letter-spacing artefacts in the source text are still present. The model handles them in reasoning but they remain noisy. Same Week 8 fix as Q&A.
- All four cases used Llama 3.3 70B at temperature 0.0. Determinism is high but not guaranteed across model upgrades. Week 8 will lock the model version in the README and methodology page.