"""
Aegis: an EU AI Act readiness tool for Irish SMEs.

Week 1: foundation only. The retrieval, classification, and Q&A features
arrive from Week 2 onwards.

Copyright (c) 2026 Noble Chidera Onyema. All Rights Reserved.
"""

import streamlit as st

st.set_page_config(
    page_title="Aegis",
    layout="wide",
)

st.title("Aegis")
st.caption("An EU AI Act readiness tool for Irish small and medium businesses.")

st.markdown(
    """
    Aegis is being built in public over the next ten weeks. When it is live,
    it will take a plain-language description of an AI system and return
    the EU AI Act obligations that apply to it, with citations to the
    relevant articles.

    The Act's main enforcement powers take effect on **2 August 2026**.

    **What you are seeing now:** the project shell. No features yet.
    **What you will see in late August:** the full tool, deployed in the EU.

    Aegis is decision-support, not legal advice. Verify every output with
    qualified counsel.
    """
)

st.divider()
st.caption(
    "(c) 2026 Noble Chidera Onyema. All Rights Reserved. "
    "Built as part of an MSc Applied AI and UX portfolio at Abertay University."
)