"""The decision loop — the heart of the project.

Two pipelines live here so the eval can compare them on identical inputs; the
ONLY difference is the uncertainty layer:

    basic_rag(question)       retrieve -> generate. Never abstains. [Stage 2]
    abstaining_rag(question)  retrieve -> gate -> generate -> support-check
                              -> (search again) -> answer / abstain. [Stage 3]

Both return the same Result shape so the UI and eval treat them uniformly.
"""

from dataclasses import dataclass, field

from config import TOP_K
from src import llm, retriever


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


# abstaining_rag() arrives in Stage 3 — it will reuse retriever.search and
# llm.generate above, then add the retrieval gate + llm.support_check.
