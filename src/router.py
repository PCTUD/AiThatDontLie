"""
The decision loop — heart of the project.

Two pipelines live here so the eval can compare them on identical inputs; the
ONLY difference is the uncertainty layer:

    basic_rag(question)       retrieve -> generate. Never abstains. [Stage 2]
    abstaining_rag(question)  retrieve -> gate -> generate -> support-check
                              -> (search again) -> answer / abstain. [Stage 3]

Both return the same Result shape so the UI and eval treat them uniformly.
"""

from dataclasses import dataclass, field

from config import ENABLE_SELF_CONSISTENCY, TOP_K, TOP_K_RETRY
from src import llm, retriever, uncertainty


@dataclass
class Result:
    """Uniform return type for both pipelines."""
    decision: str                       # "ANSWER" | "SEARCH_AGAIN" | "ABSTAIN"
    answer: str                         # the answer text, or the abstention message
    hits: list[dict] = field(default_factory=list)   # retrieved passages (for citations/UI)
    used: list[int] = field(default_factory=list)     # passage numbers the model cited
    reason: str = ""                    # why it abstained / what the support check said
    signals: dict = field(default_factory=dict)       # raw signal values, for the eval & debugging


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — plain RAG, the baseline. No gating, no second-guessing.
# ─────────────────────────────────────────────────────────────────────────────
def basic_rag(question: str) -> Result:
    hits = retriever.search(question, k=TOP_K)
    if not hits:
        # Only happens on an empty index; basic RAG otherwise always answers.
        return Result(decision="ANSWER", answer="(no passages found)", hits=[])
    gen = llm.generate(question, hits)
    return Result(
        decision="ANSWER",
        answer=gen["answer"],
        hits=hits,
        used=gen["used"],
        signals={"top_score": hits[0]["score"]},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — abstaining RAG. Same retrieve+generate core, wrapped in the
# uncertainty layer. This is the whole point of the project.
# ─────────────────────────────────────────────────────────────────────────────
def _abstain(reason: str, hits: list[dict], signals: dict) -> Result:
    return Result(
        decision="ABSTAIN",
        answer="I don't have enough evidence in the corpus to answer this confidently.",
        hits=hits,
        reason=reason,
        signals=signals,
    )


def abstaining_rag(question: str) -> Result:
    """retrieve -> gate -> generate -> support-check -> (search again) -> answer/abstain.

    Decision flow:
      1. SIGNAL 1 (retrieval gate): if nothing relevant retrieved, ABSTAIN now
         (no LLM token spent).
      2. Generate a grounded answer.
      3. SIGNAL 2 (support check): if the answer isn't backed by the passages,
         try ONE wider retrieval (SEARCH_AGAIN); if it still isn't, ABSTAIN.
      4. (SIGNAL 3, optional) self-consistency — Stage 5.
      5. Otherwise ANSWER.
    """
    signals: dict = {}

    # --- SIGNAL 1: retrieval gate -------------------------------------------
    hits = retriever.search(question, k=TOP_K)
    passed, top_score = uncertainty.retrieval_gate(hits)
    signals["top_score"] = top_score
    if not passed:
        return _abstain(
            f"Retrieval gate: best passage scored {top_score:.2f} < threshold; "
            "nothing relevant in the corpus.",
            hits,
            signals,
        )

    # --- Generate, then SIGNAL 2: support check -----------------------------
    gen = llm.generate(question, hits)
    sup = uncertainty.support(question, gen["answer"], hits)
    signals["support"] = sup["label"]

    searched_again = False
    if not sup["is_supported"]:
        # SEARCH_AGAIN: one wider retrieval pass before giving up. A larger k
        # can surface a passage the first pass missed; cheap insurance against
        # a too-narrow first retrieval.
        searched_again = True
        hits = retriever.search(question, k=TOP_K_RETRY)
        gen = llm.generate(question, hits)
        sup = uncertainty.support(question, gen["answer"], hits)
        signals["support_retry"] = sup["label"]
        signals["searched_again"] = True
        if not sup["is_supported"]:
            return _abstain(
                f"Support check: answer not grounded in the passages "
                f"({sup['label']}) even after searching again. {sup['reason']}",
                hits,
                signals,
            )

    # --- SIGNAL 3: self-consistency (optional, Stage 5) ---------------------
    if ENABLE_SELF_CONSISTENCY:
        disagrees, sim = uncertainty.disagrees_across_subsets(question, hits)
        signals["consistency_sim"] = sim
        if disagrees:
            return _abstain(
                f"Self-consistency: answers from different evidence subsets "
                f"disagreed (similarity {sim:.2f}).",
                hits,
                signals,
            )

    return Result(
        decision="SEARCH_AGAIN" if searched_again else "ANSWER",
        answer=gen["answer"],
        hits=hits,
        used=gen["used"],
        reason=sup["reason"],
        signals=signals,
    )
