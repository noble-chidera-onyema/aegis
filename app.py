"""
Ugegbe - EU AI Act readiness for small businesses, Ireland-focused.

Streamlit front end (Week 6). Four screens wired to the Python core built in
Weeks 2 to 5:

  1. Inventory   - the user describes an AI system
  2. Classify    - calls classify_system(), shows tier, confidence, reasoning, citations
  3. Obligations - calls generate_report(), shows the applicable Articles with verifiable citations
  4. Ask         - calls grounded_qa() for questions about the Act

Design: see docs/DESIGN_BRIEF.md (institutional / regulatory-document aesthetic).

Data handling (binding, see spec Section 6): session-state only. Nothing is
written to disk, nothing is logged server-side. When the browser tab closes,
the data is gone. Inputs are sent to Groq for processing; the privacy warning
next to every input field says so.

This is decision-support, not legal advice.

Copyright (c) 2026 Noble Chidera Onyema. All Rights Reserved.
"""

from __future__ import annotations

import logging
import time
import traceback

import streamlit as st

# --- Logging (developer-facing, console only) ------------------------------
# Full tracebacks go to the console so that during development the exact
# location of any failure is visible. No user input is logged. In production
# (Week 9) this is where a Sentry handler would attach.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("aegis.app")


# --- Page config -----------------------------------------------------------
st.set_page_config(
    page_title="Ugegbe - EU AI Act Readiness",
    page_icon="\u2696",  # scales of justice, rendered by the browser, not an emoji asset
    layout="centered",
    initial_sidebar_state="collapsed",
)


# --- Cached backend resources ----------------------------------------------
# st.cache_resource loads these ONCE per server process and shares them across
# all user sessions. Without this, every user (and every rerun) would reload
# the embedding model and re-open the Chroma client, which is the single
# biggest cause of a sluggish Streamlit app under concurrent use. This is the
# main concurrency measure for the demo deploy.
#
# The imports are inside the cached functions so that a missing backend module
# surfaces as a handled error in the UI rather than crashing the whole app at
# import time.

@st.cache_resource(show_spinner=False)
def get_classifier():
    """Import and return the classify_system callable."""
    from src.aegis.classify import classify_system
    return classify_system


@st.cache_resource(show_spinner=False)
def get_report_generator():
    """Import and return the generate_report callable."""
    from src.aegis.obligations import generate_report
    return generate_report

@st.cache_resource(show_spinner=False)
def get_index():
    """Build or load the EU AI Act index ONCE per process, shared by all calls."""
    from src.aegis.classify import load_index
    return load_index()

@st.cache_resource(show_spinner=False)
def get_qa():
    """Return a question -> answer wrapper over the Week 3 grounded-QA pipeline.

    The grounded_qa module exposes lower-level pieces rather than a single
    entry point. The real flow is: build a Groq client, load the index once,
    retrieve chunks for the question, then ask the model with those chunks.
    The client and index are loaded a single time and closed over here so the
    wrapper only does per-question work (retrieve + ask)."""
    from src.aegis.grounded_qa import Groq, retrieve_for_question, ask, TOP_K

    client = Groq()
    index = get_index()

    def answer_question(question: str) -> str:
        nodes = retrieve_for_question(index, question, top_k=TOP_K)
        return ask(client, question, nodes)

    return answer_question


# --- Styling ---------------------------------------------------------------
def inject_styles() -> None:
    """Inject the institutional design tokens as CSS. See docs/DESIGN_BRIEF.md."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #FBFAF7;
            --text: #1A1A1A;
            --accent: #1B3A5C;
            --border: #D8D4CC;
            --tier-prohibited: #8B2635;
            --tier-high: #8A5E15;
            --tier-limited: #3A6B8C;
            --tier-minimal: #2D5A3D;
            --banner-bg: #FBF3DF;
            --banner-border: #C9A227;
        }

        /* App background and base font */
        .stApp {
            background-color: var(--bg);
        }
        html, body, [class*="css"] {
            font-family: 'Source Sans 3', sans-serif;
            color: var(--text);
        }

        /* Hide Streamlit's default top toolbar and footer chrome.
           The dark bar with the Deploy button was clipping our header. */
        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }
        [data-testid="stToolbar"] { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Base body paragraphs: readable weight and full contrast */
        .stApp p, .stApp li, .stApp label, .stMarkdown p {
            color: var(--text);
            font-weight: 400;
            font-size: 1rem;
            line-height: 1.55;
        }

        /* Headings in serif */
        h1, h2, h3, h4 {
            font-family: 'Lora', serif !important;
            color: var(--accent) !important;
            letter-spacing: -0.01em;
        }

        /* Constrain the main column width for a document feel */
        .block-container {
            max-width: 820px;
            padding-top: 2.5rem;
        }

        /* Header strip */
        .aegis-header {
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.6rem;
            margin-bottom: 0.4rem;
        }
        .aegis-wordmark {
            font-family: 'Lora', serif;
            font-weight: 700;
            font-size: 1.7rem;
            color: var(--accent);
            letter-spacing: 0.02em;
        }
        .aegis-tagline {
            font-family: 'Source Sans 3', sans-serif;
            font-size: 0.85rem;
            color: #5A5A5A;
            margin-top: -0.2rem;
        }

        /* Disclaimer banner */
        .aegis-disclaimer {
            background-color: var(--banner-bg);
            border: 1px solid var(--banner-border);
            border-radius: 4px;
            padding: 0.5rem 0.8rem;
            font-size: 0.85rem;
            color: #5A4A12;
            margin-bottom: 1.2rem;
        }

        /* Human-in-the-loop escalation banner. Stronger than the standing
           disclaimer: heavier left border in the high-risk ochre, used only
           when the classification is flagged for review. */
        .aegis-review-banner {
            background-color: #FBF3E3;
            border: 1px solid var(--tier-high);
            border-left: 4px solid var(--tier-high);
            border-radius: 4px;
            padding: 0.7rem 1rem;
            font-size: 0.9rem;
            color: #4A3608;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }
        .aegis-review-banner strong { color: #3A2A06; }

        /* Small state badge that carries the human's review decision across
           screens (acknowledged / disputed). */
        .aegis-review-state {
            display: inline-block;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 0.25rem 0.65rem;
            border-radius: 3px;
            margin-bottom: 1rem;
        }
        .aegis-review-state.acknowledged {
            background-color: #E8F0E8;
            color: var(--tier-minimal);
            border: 1px solid var(--tier-minimal);
        }
        .aegis-review-state.disputed {
            background-color: #F4E6E8;
            color: var(--tier-prohibited);
            border: 1px solid var(--tier-prohibited);
        }

        /* Privacy warning under inputs */
        .aegis-privacy {
            font-size: 0.85rem;
            color: #4A4A4A;
            border-left: 3px solid var(--banner-border);
            padding-left: 0.7rem;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
            line-height: 1.5;
        }

        /* Result card with tier colour bar */
        .aegis-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-left-width: 6px;
            border-radius: 6px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1rem;
            animation: aegisFade 0.5s ease-out both;
        }
        .tier-prohibited { border-left-color: var(--tier-prohibited); }
        .tier-high       { border-left-color: var(--tier-high); }
        .tier-limited    { border-left-color: var(--tier-limited); }
        .tier-minimal    { border-left-color: var(--tier-minimal); }

        .tier-label {
            font-family: 'Lora', serif;
            font-weight: 700;
            font-size: 1.25rem;
        }
        .tier-label.tier-prohibited { color: var(--tier-prohibited); }
        .tier-label.tier-high       { color: var(--tier-high); }
        .tier-label.tier-limited    { color: var(--tier-limited); }
        .tier-label.tier-minimal    { color: var(--tier-minimal); }

        /* Citation / page references in monospace */
        .aegis-cite {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            color: var(--accent);
            background: #F1EFEA;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
        }

        /* Obligation block */
        .aegis-obligation {
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
            background: #FFFFFF;
            animation: aegisFade 0.5s ease-out both;
        }
        .aegis-obligation h4 {
            margin-bottom: 0.3rem;
        }
        .aegis-ai-note {
            background: #F4F6F8;
            border-left: 3px solid var(--tier-limited);
            padding: 0.6rem 0.9rem;
            font-size: 0.92rem;
            color: #1A1A1A !important;
            line-height: 1.5;
            margin-top: 0.6rem;
            border-radius: 3px;
        }
        .aegis-ai-note, .aegis-ai-note * {
            color: #1A1A1A !important;
        }
        .aegis-ai-note-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #6A6A6A !important;
            font-weight: 600;
        }

        /* Staggered fade for sequential cards */
        @keyframes aegisFade {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* Primary buttons in accent colour */
        .stButton > button {
            background-color: var(--accent);
            color: #FFFFFF;
            border: none;
            border-radius: 4px;
            font-family: 'Source Sans 3', sans-serif;
            font-weight: 600;
        }
        .stButton > button:hover {
            background-color: #25496F;
            color: #FFFFFF;
        }

        /* Footer */
        .aegis-footer {
            border-top: 1px solid var(--border);
            margin-top: 2.5rem;
            padding-top: 0.8rem;
            font-size: 0.82rem;
            color: #5A5A5A;
            line-height: 1.5;
        }

        /* Text input and textarea: force light background, dark readable text.
           Streamlit's dark-mode default was leaking a near-black box onto
           our warm off-white theme. */
        .stTextArea textarea,
        .stTextInput input {
            background-color: #FFFFFF !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-family: 'Source Sans 3', sans-serif !important;
            font-size: 1rem !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #8A8A8A !important;
            opacity: 1;
        }
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        /* Chat input box */
        [data-testid="stChatInput"] textarea {
            background-color: #FFFFFF !important;
            color: var(--text) !important;
        }

        /* Chat message bubbles: Streamlit's defaults render with their own
           grey/coloured backgrounds and faint text on our theme. Force a
           clean light card with dark, readable text for both roles. */
        [data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            padding: 0.8rem 1rem !important;
            margin-bottom: 0.6rem !important;
        }
        [data-testid="stChatMessage"] * {
            color: #1A1A1A !important;
        }
        [data-testid="stChatMessage"] p {
            color: #1A1A1A !important;
            font-weight: 400 !important;
            line-height: 1.55 !important;
        }

        /* Streamlit alert boxes (warning, info, success): readable contrast
           on our theme instead of the pale default. */
        [data-testid="stAlert"] {
            color: #1A1A1A !important;
        }
        [data-testid="stAlert"] p {
            color: #1A1A1A !important;
            font-weight: 500 !important;
        }

        /* Tabs: make the labels readable and weight the active one */
        .stTabs [data-baseweb="tab"] {
            font-family: 'Source Sans 3', sans-serif;
            font-weight: 600;
            color: #5A5A5A;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }

        /* Captions a touch darker for legibility */
        .stCaption, [data-testid="stCaptionContainer"] {
            color: #6A6A6A !important;
        }

        /* --- Expander: make the toggle obvious and accessible -------------
           Streamlit's default expander caret is faint and only clear on
           hover. This makes the whole header read as a clickable control. */
        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 1.2rem;
            background-color: #FFFFFF;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details > summary {
            font-family: 'Source Sans 3', sans-serif;
            font-weight: 600;
            font-size: 0.95rem;
            color: var(--accent);
            padding: 0.7rem 1rem;
            cursor: pointer;
        }
        /* Force the caret/marker to be visible at all times, not just hover */
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
            color: var(--accent) !important;
            fill: var(--accent) !important;
            opacity: 1 !important;
            width: 1.1rem !important;
            height: 1.1rem !important;
        }
        [data-testid="stExpander"] summary:hover {
            color: #142b44;
            background-color: #F4F1EA;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            padding: 0.2rem 1rem 0.8rem 1rem;
        }

        /* --- Header brand mark -------------------------------------------- */
        .aegis-header {
            display: flex;
            align-items: center;
            gap: 0.7rem;
        }
        .aegis-brandmark {
            flex: 0 0 auto;
            line-height: 0;
        }
        .aegis-headtext { display: flex; flex-direction: column; }

        /* --- Spacing / hierarchy polish ----------------------------------- */
        .block-container { padding-top: 2.2rem; }
        h3, .stSubheader { margin-top: 0.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Header strip and the always-visible disclaimer banner."""
    st.markdown(
        """
        <div class="aegis-header">
            <div class="aegis-brandmark">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none"
                     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <path d="M12 2.2 4 5.1v6.2c0 4.6 3.2 8.4 8 10.5 4.8-2.1 8-5.9 8-10.5V5.1L12 2.2Z"
                          fill="#1B3A5C"/>
                    <path d="M12 4.3 6 6.5v4.8c0 3.5 2.4 6.5 6 8.2 3.6-1.7 6-4.7 6-8.2V6.5L12 4.3Z"
                          fill="#FBFAF7"/>
                    <path d="M8.6 12.1l2.3 2.3 4.5-4.6" stroke="#1B3A5C" stroke-width="1.6"
                          stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                </svg>
            </div>
            <div class="aegis-headtext">
                <div class="aegis-wordmark">Ugegbe</div>
                <div class="aegis-tagline">See your AI system the way the EU AI Act sees it</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="aegis-disclaimer">
            <strong>Ugegbe is decision-support, not legal advice.</strong>
            It helps you understand which parts of the EU AI Act may apply to
            your systems. Verify any conclusion with qualified counsel before
            acting on it.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="aegis-footer">
            Ugegbe is an independent portfolio project. It is not affiliated with
            the European Commission, the AI Office, or any Irish authority. Ugegbe
            stores nothing; see "Privacy: what happens to what you type" on the
            Inventory tab for how your inputs are handled. Do not enter personal data.
            &copy; 2026 Noble Chidera Onyema. All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Session state ---------------------------------------------------------
def init_state() -> None:
    """Initialise shared session state. Survives reruns; cleared on tab close."""
    defaults = {
        "system_description": "",
        "classification": None,   # holds the Classification dataclass after classify
        "report": None,           # holds the ObligationsReport after generate
        "chat_history": [],        # list of (role, text) tuples for the Ask screen
        "last_error": None,        # for the error-boundary fallback
        "review_state": "unreviewed",  # human-in-the-loop: unreviewed | acknowledged | disputed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_workflow() -> None:
    """Clear the workflow state but keep the app running. Used by the reset button."""
    st.session_state.system_description = ""
    st.session_state.classification = None
    st.session_state.report = None
    st.session_state.last_error = None
    st.session_state.review_state = "unreviewed"


# --- Error boundary helper -------------------------------------------------
def run_safely(label: str, fn, *args, **kwargs):
    """
    Run a backend call inside an error boundary.

    On success: returns (result, None).
    On failure: logs the full traceback to the console (developer-facing),
    returns (None, friendly_message). The caller renders the friendly message
    and a reset button. The user's typed inputs are never lost because they
    live in session_state, not in the call.
    """
    try:
        result = fn(*args, **kwargs)
        return result, None
    except Exception as exc:  # noqa: BLE001  (we genuinely want to catch everything here)
        # Developer-facing: exact location and type in the console.
        logger.error("Failure in %s: %s\n%s", label, exc, traceback.format_exc())
        # User-facing: calm, no internals leaked.
        friendly = (
            "Something went wrong while processing that. Your inputs are saved. "
            "Please try again. If it keeps happening, the language model service "
            "may be temporarily unavailable."
        )
        return None, friendly


def show_error_fallback(message: str, key: str = "reset") -> None:
    """Render the friendly error state with a reset button. No progress is lost.

    The key argument makes the reset button unique per screen, so Streamlit
    does not raise a duplicate-element-id error when two screens both render
    the fallback.
    """
    st.markdown(
        f"""
        <div class="aegis-card tier-high">
            <div class="tier-label tier-high">Something went wrong</div>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Reset and try again", key=f"reset_{key}"):
        st.session_state.last_error = None
        st.rerun()


# --- Small UI helpers ------------------------------------------------------
TIER_CLASS = {
    "prohibited": "tier-prohibited",
    "high-risk": "tier-high",
    "high": "tier-high",
    "limited-risk": "tier-limited",
    "limited": "tier-limited",
    "minimal-risk": "tier-minimal",
    "minimal": "tier-minimal",
}


def tier_css_class(tier: str) -> str:
    """Map a tier string to its CSS class, defaulting to limited if unknown."""
    return TIER_CLASS.get(tier.strip().lower(), "tier-limited")


def skeleton_pause(seconds: float = 0.5) -> None:
    """A short, deliberate pause so results resolve smoothly rather than snapping in."""
    time.sleep(seconds)


PRIVACY_WARNING = (
    '<div class="aegis-privacy">Describe what the system does, not who uses '
    'it. Do not enter personal data (names, emails, anything identifying a '
    'person). Your text is sent to a third-party model (Groq) to generate the '
    'answer and is not stored by Ugegbe.</div>'
)


# --- Screen 1: Inventory ---------------------------------------------------
def screen_inventory() -> None:
    st.subheader("Describe your AI system")
    st.write(
        "Describe one system in plain language. Ugegbe classifies its EU AI Act "
        "risk tier and lists the obligations that apply."
    )
    with st.expander("How this works"):
        st.markdown(
"""- Describe **what the system does**, not who uses it. One system at a time.
- Ugegbe reads your description against the Act and assigns a risk tier (prohibited, high-risk, limited-risk, or minimal-risk).
- The Classification tab shows the tier and the reasoning. The Obligations tab lists what the tier requires, with page citations you can verify against the Act.
- This is decision-support, not legal advice."""
        )

    with st.expander("Privacy: what happens to what you type"):
        st.markdown(
"""- **Ugegbe stores nothing.** What you type lives in your browser session only and is gone when you close the tab. There is no account, no database, no server-side record kept by Ugegbe.
- **Your text is sent to Groq** to generate the classification and answers. Per Groq's policy, inference requests are **not retained by default**; they may be logged briefly (up to 30 days) only when investigating an error or suspected abuse.
- **In transit and at rest, data is encrypted** (TLS 1.2+ and AES-256), per Groq.
- **Do not enter personal data**, names, emails, or anything identifying a person. Describe what the system does, not who uses it. You do not need personal data to get a useful classification.
- Ugegbe is an independent project and is not affiliated with the European Commission, the AI Office, or any Irish authority."""
        )

    description = st.text_area(
        "What does the system do?",
        value=st.session_state.system_description,
        height=160,
        placeholder=(
            "Example: An AI tool that automatically ranks job candidates by "
            "scoring their CVs for fit with open roles. A human recruiter "
            "reviews the shortlist and makes the final interview decisions."
        ),
        label_visibility="collapsed",
    )
    st.markdown(PRIVACY_WARNING, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        classify_clicked = st.button("Classify this system")
    with col2:
        if st.session_state.classification is not None:
            st.caption("Classified. See the Classification tab.")

    if classify_clicked and not st.session_state.get("classify_running", False):
        if not description.strip():
            st.warning("Please describe the system before classifying.")
            return
        st.session_state.system_description = description.strip()
        st.session_state.report = None  # invalidate any prior report

        st.session_state.classify_running = True
        classify_system = get_classifier()
        with st.spinner("Reading the Act and classifying the system..."):
            skeleton_pause(0.5)
            result, err = run_safely(
                "classify_system", classify_system, st.session_state.system_description,
                None, get_index()
            )
        if err:
            st.session_state.last_error = err
        else:
            st.session_state.classification = result
            st.session_state.last_error = None
            # A new classification invalidates any prior human review and any
            # report built on the old tier. Reset both so review status never
            # leaks from one system to the next.
            st.session_state.review_state = "unreviewed"
            st.session_state.report = None
            st.success("Classification complete. Open the Classification tab to see the result.")
        st.session_state.classify_running = False

    if st.session_state.last_error:
        show_error_fallback(st.session_state.last_error, key="inventory")


# --- Screen 2: Classification ----------------------------------------------
def screen_classification() -> None:
    st.subheader("Classification result")

    clf = st.session_state.classification
    if clf is None:
        st.info("No classification yet. Describe a system on the Inventory tab and classify it.")
        return

    st.write(
        "Ugegbe assigned this risk tier and explained why. Read it, then accept "
        "it or flag your disagreement below."
    )
    with st.expander("How to read this"):
        st.markdown(
"""- **Tier** is the EU AI Act risk category Ugegbe assigned to your system.
- **Confidence** is how strongly the retrieved text supported that tier.
- **Cited Articles** point to where in the Act the reasoning comes from.
- An automated tier can be wrong even when it looks confident. You stay in control: accept the tier, or flag disagreement, before generating the obligations report."""
        )

    # The Classification dataclass fields: tier, confidence, reasoning,
    # citations (list), needs_human_review (bool). Access defensively so a
    # change in the dataclass does not crash the UI.
    tier = getattr(clf, "tier", "unknown")
    confidence = getattr(clf, "confidence", "unknown")
    reasoning = getattr(clf, "reasoning", "")
    citations = getattr(clf, "citations", []) or []
    needs_review = getattr(clf, "needs_human_review", False)

    css = tier_css_class(tier)

    st.markdown(
        f"""
        <div class="aegis-card {css}">
            <span class="tier-label {css}">{tier}</span>
            &nbsp;&nbsp;<span style="color:#6A6A6A;">confidence: {confidence}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Why this tier**")
    st.write(reasoning)

    if citations:
        st.markdown("**Cited Articles**")
        cite_html = " ".join(
            f'<span class="aegis-cite">{str(c)}</span>' for c in citations
        )
        st.markdown(cite_html, unsafe_allow_html=True)

    # --- Human-in-the-loop (Week 7) ----------------------------------------
    # Option B, the design aligned with EU AI Act Article 14 on human
    # oversight and automation bias: a standing caution shows on EVERY
    # classification (the floor), and a stronger banner escalates when the
    # model flags the case for review. The flag is known to underfire on
    # borderline cases (see Week 4 notes), so the standing floor guarantees
    # the user is never left with zero caution when the model is overconfident.

    # Escalation banner: only when the model flagged the case.
    if needs_review:
        st.markdown(
            """
            <div class="aegis-review-banner">
                <strong>Flagged for human review.</strong> This case is
                ambiguous, near a tier boundary, or only weakly supported by
                the retrieved text. Do not act on this tier without checking it
                against the Act and qualified counsel.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Standing caution floor: on every classification, flagged or not.
    st.markdown(
        """
        <div class="aegis-disclaimer" style="margin-top:0.8rem;">
            Ugegbe is decision-support, not legal advice. An automated tier can
            be wrong even when it looks confident. Confirm this classification
            against the Act and with qualified counsel before relying on it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Review-and-edit controls: capture the human's judgement. This is the
    # "loop" in human-in-the-loop, the user records that they reviewed the
    # tier, or that they disagree with it. The decision carries to the
    # Obligations screen via session state.
    current = st.session_state.get("review_state", "unreviewed")
    if current == "acknowledged":
        st.markdown(
            '<div class="aegis-review-state acknowledged">You reviewed and accepted this classification</div>',
            unsafe_allow_html=True,
        )
    elif current == "disputed":
        st.markdown(
            '<div class="aegis-review-state disputed">You flagged disagreement with this classification</div>',
            unsafe_allow_html=True,
        )

    col_ack, col_dispute = st.columns(2)
    with col_ack:
        if st.button("I have reviewed and accept this tier", key="hitl_ack"):
            st.session_state.review_state = "acknowledged"
            st.rerun()
    with col_dispute:
        if st.button("I disagree with this tier", key="hitl_dispute"):
            st.session_state.review_state = "disputed"
            st.rerun()

    st.divider()
    st.write("Generate the full obligations report for this system:")
    if st.button("View obligations") and not st.session_state.get("obligations_running", False):
        st.session_state.obligations_running = True
        generate_report = get_report_generator()
        with st.spinner("Mapping obligations and retrieving source passages..."):
            skeleton_pause(0.5)
            report, err = run_safely(
                "generate_report",
                generate_report,
                clf,
                st.session_state.system_description,
                None,
                get_index(),
            )
        if err:
            st.session_state.last_error = err
        else:
            st.session_state.report = report
            st.session_state.last_error = None
            st.success("Obligations report ready. Open the Obligations tab.")
        st.session_state.obligations_running = False

    if st.session_state.last_error:
        show_error_fallback(st.session_state.last_error, key="classification")


# --- Screen 3: Obligations -------------------------------------------------
def screen_obligations() -> None:
    st.subheader("Obligations and gap report")

    report = st.session_state.report
    if report is None:
        st.info(
            "No report yet. Classify a system, then click 'View obligations' on "
            "the Classification tab."
        )
        return

    st.write(
        "The Articles that apply to this tier, each with the requirement, a "
        "note on why it applies to your system, and a checklist question."
    )
    with st.expander("How to read this"):
        st.markdown(
"""- Each card is one operative Article of the Act that applies to this risk tier.
- The **page citation** points to where that Article begins in the source PDF, so you can verify it against the Act yourself.
- The **AI-generated note** is written for your specific system and is clearly labelled. The Article text and citations are not AI-generated.
- The **checklist question** is what to ask yourself to judge whether you already meet the obligation.
- This is decision-support, not legal advice. Confirm with qualified counsel before acting."""
        )

    tier = getattr(report, "tier", "unknown")
    obligations = getattr(report, "obligations", []) or []
    notes = getattr(report, "notes", []) or []
    css = tier_css_class(tier)

    st.markdown(
        f"""
        <div class="aegis-card {css}">
            <span class="tier-label {css}">{tier}</span>
            &nbsp;&nbsp;<span style="color:#6A6A6A;">{len(obligations)} obligation(s)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Carry the human's review decision from the Classification screen, so the
    # report reflects whether a person reviewed or disputed the tier it rests on.
    review_state = st.session_state.get("review_state", "unreviewed")
    if review_state == "acknowledged":
        st.markdown(
            '<div class="aegis-review-state acknowledged">You reviewed and accepted the classification this report is based on</div>',
            unsafe_allow_html=True,
        )
    elif review_state == "disputed":
        st.markdown(
            '<div class="aegis-review-state disputed">You flagged disagreement with the classification this report is based on. Treat these obligations as provisional.</div>',
            unsafe_allow_html=True,
        )

    if notes:
        for note in notes:
            st.markdown(
                f'<div class="aegis-disclaimer">{note}</div>',
                unsafe_allow_html=True,
            )

    for i, ob in enumerate(obligations, start=1):
        article = getattr(ob, "article", "")
        title = getattr(ob, "title", "")
        description = getattr(ob, "description", "")
        checklist_q = getattr(ob, "checklist_question", "")
        compliance_level = getattr(ob, "compliance_level", "")
        source_page = getattr(ob, "source_page", None)
        applies_because = getattr(ob, "applies_because", "")

        if source_page is not None:
            page_html = f'<span class="aegis-cite">{article}, page {source_page}</span>'
        else:
            page_html = f'<span class="aegis-cite">{article}</span>'

        # The note block is assembled as a single line with no leading
        # indentation. Streamlit's markdown renderer treats indented HTML as
        # a code block and escapes it, which is why an earlier version showed
        # the raw div tags. Keeping the whole card on flat single lines avoids
        # that entirely.
        if applies_because:
            note_block = (
                '<div class="aegis-ai-note">'
                '<div class="aegis-ai-note-label">AI-generated note for your system</div>'
                f'{applies_because}'
                '</div>'
            )
        else:
            note_block = ""

        card_html = (
            '<div class="aegis-obligation">'
            f'<h4>{article}: {title}</h4>'
            f'{page_html}'
            f'&nbsp;<span style="color:#6A6A6A;font-size:0.85rem;">{compliance_level}</span>'
            f'<p style="margin-top:0.6rem;">{description}</p>'
            f'{note_block}'
            f'<p style="margin-top:0.6rem;"><strong>Checklist:</strong> {checklist_q}</p>'
            '</div>'
        )

        st.markdown(card_html, unsafe_allow_html=True)

    st.divider()
    st.caption(
        "Every page reference points to where that Article begins in the source "
        "PDF of the Act, so you can verify each one. The per-system notes are "
        "AI-generated and clearly labelled."
    )


# --- Screen 4: Ask ---------------------------------------------------------
def screen_ask() -> None:
    st.subheader("Ask about the Act")
    st.write(
        "Ask a question about the EU AI Act. Answers quote and cite the actual "
        "legislation, and Ugegbe declines when the text does not support an answer."
    )
    with st.expander("How this works and what to ask"):
        st.markdown(
'''- Answers are grounded in the text of the Act and cite the page each claim comes from. If the retrieved text does not support an answer, Ugegbe says so rather than guessing.
- Questions that name an Article ("What does Article 13 require?") are answered from that Article directly.
- Broader questions ("What makes a system high-risk?") are answered from the most relevant passages found.
- This is decision-support, not legal advice.'''
        )

    # Replay the chat history.
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("e.g. What does Article 13 require?")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        grounded_qa = get_qa()
        with st.chat_message("assistant"):
            with st.spinner("Searching the Act and composing a grounded answer..."):
                skeleton_pause(0.4)
                answer, err = run_safely("grounded_qa", grounded_qa, prompt)
            if err:
                st.markdown(
                    "Something went wrong answering that. Please try again. Your "
                    "earlier messages are still here."
                )
                logger.info("grounded_qa error surfaced to user")
            else:
                st.markdown(answer)
                st.session_state.chat_history.append(("assistant", answer))

    st.markdown(PRIVACY_WARNING, unsafe_allow_html=True)


# --- Main ------------------------------------------------------------------
def main() -> None:
    inject_styles()
    init_state()
    render_header()

    tab_inventory, tab_classify, tab_obligations, tab_ask = st.tabs(
        ["Inventory", "Classification", "Obligations", "Ask"]
    )
    with tab_inventory:
        screen_inventory()
    with tab_classify:
        screen_classification()
    with tab_obligations:
        screen_obligations()
    with tab_ask:
        screen_ask()

    render_footer()


if __name__ == "__main__":
    main()