"""Minimal demo UI.

    streamlit run app.py

Stage 2: type a question, get a cited answer (basic RAG only).
Stage 3 will add a pipeline toggle and a colored decision badge.
"""

import streamlit as st

from src import router

st.set_page_config(page_title="RAG that knows when to shut up", page_icon="🎾")
st.title("🎾 RAG that knows when to shut up")
st.caption("Ask about tennis. The corpus is ~24 Wikipedia articles.")

question = st.text_input(
    "Question",
    placeholder="e.g. How does a tiebreak work?",
)

if question:
    with st.spinner("Retrieving and generating..."):
        result = router.basic_rag(question)

    st.markdown("### Answer")
    st.write(result.answer)

    if result.hits:
        top = result.signals.get("top_score")
        with st.expander(
            f"Sources & retrieval (top similarity: {top:.2f})"
            if top is not None else "Sources & retrieval"
        ):
            for i, h in enumerate(result.hits, 1):
                cited = " ✅ cited" if i in result.used else ""
                st.markdown(
                    f"**[{i}] {h['source']}** — score {h['score']:.3f}{cited}"
                )
                st.caption(h["text"][:300] + "...")
