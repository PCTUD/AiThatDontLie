"""Minimal demo UI.

    streamlit run app.py

Pick a pipeline (basic vs abstaining), ask a question, and see the decision:
  ANSWER (green) · SEARCH_AGAIN (amber) · ABSTAIN (red),
plus the answer, the reason, and the retrieved sources with scores + citations.
"""

import streamlit as st

from src import router

st.set_page_config(page_title="RAG that knows when to shut up", page_icon="🎾")
st.title("🎾 RAG that knows when to shut up")
st.caption("Ask about tennis. The corpus is ~24 Wikipedia articles.")

pipeline = st.radio(
    "Pipeline",
    ["Abstaining RAG", "Basic RAG (baseline)"],
    horizontal=True,
    help="Basic always answers. Abstaining can say 'I don't know' when the "
    "evidence doesn't support an answer.",
)

question = st.text_input("Question", placeholder="e.g. How does a tiebreak work?")

# Colors for the decision badge.
BADGE = {
    "ANSWER": ("🟢 ANSWER", "#1a7f37"),
    "SEARCH_AGAIN": ("🟡 SEARCH AGAIN → answered", "#9a6700"),
    "ABSTAIN": ("🔴 ABSTAIN", "#cf222e"),
}

if question:
    run = router.basic_rag if pipeline.startswith("Basic") else router.abstaining_rag
    with st.spinner("Thinking..."):
        result = run(question)

    label, color = BADGE.get(result.decision, (result.decision, "#57606a"))
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-weight:600;font-size:0.9em'>{label}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("### Answer")
    st.write(result.answer)

    if result.reason and result.decision != "ANSWER":
        st.info(f"**Why:** {result.reason}")

    if result.hits:
        top = result.signals.get("top_score")
        header = "Sources & signals"
        if top is not None:
            header += f"  (top similarity: {top:.2f})"
        with st.expander(header):
            if result.signals:
                st.caption("Signals: " + ", ".join(
                    f"{k}={v}" for k, v in result.signals.items()
                ))
            for i, h in enumerate(result.hits, 1):
                cited = " ✅ cited" if i in result.used else ""
                st.markdown(f"**[{i}] {h['source']}** — score {h['score']:.3f}{cited}")
                st.caption(h["text"][:300] + "...")
