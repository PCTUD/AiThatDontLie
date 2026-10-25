"""Side-by-side demo UI.

    streamlit run app.py

Both pipelines run concurrently for every question. Results appear in two columns:
  Basic RAG (left) — always answers.
  Abstaining RAG (right) — can ANSWER, SEARCH_AGAIN, or ABSTAIN.

Decision badges: ANSWER (green) · SEARCH_AGAIN (amber) · ABSTAIN (red).
"""

import concurrent.futures

import streamlit as st

from src import retriever, router

st.set_page_config(page_title="RAG that knows when to shut up", layout="wide", page_icon="🎾")

# Warm up the embedding model before any threaded calls to avoid a race on the
# lazy singleton in retriever._load() when both pipelines start concurrently.
retriever.search("warmup", k=1)
st.title("🎾 RAG that knows when to shut up")
st.caption("Ask about tennis. The corpus is ~24 Wikipedia articles.")

question = st.text_input("Question", placeholder="e.g. How does a tiebreak work?")

BADGE = {
    "ANSWER": ("🟢 ANSWER", "#1a7f37"),
    "SEARCH_AGAIN": ("🟡 SEARCH AGAIN → answered", "#9a6700"),
    "ABSTAIN": ("🔴 ABSTAIN", "#cf222e"),
}


def render_result(result):
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


if question:
    with st.spinner("Running both pipelines…"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_basic = pool.submit(router.basic_rag, question)
            fut_abstain = pool.submit(router.abstaining_rag, question)
            basic_result = fut_basic.result()
            abstain_result = fut_abstain.result()

    col_basic, col_abstain = st.columns(2)

    with col_basic:
        st.subheader("Basic RAG")
        render_result(basic_result)

    with col_abstain:
        st.subheader("Abstaining RAG")
        render_result(abstain_result)
