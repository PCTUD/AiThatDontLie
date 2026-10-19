"""The uncertainty signals — the part that makes this more than plain RAG.

Each signal is a small function returning a verdict the router can act on.
Keeping them separate (not buried in router.py) means each can be unit-tested
and reasoned about on its own, and the eval can attribute an abstention to a
specific signal.

    SIGNAL 1  retrieval_gate(hits)                 -> (passed: bool, top_score)
    SIGNAL 2  support(question, answer, hits)       -> label + reason   (wraps llm.support_check)
    SIGNAL 3  disagrees_across_subsets(...)         -> (disagrees, sim)  [Stage 5]
"""

from config import (
    TAU_CONSISTENCY,
    TAU_RETRIEVE,
    TREAT_PARTIAL_AS_UNSUPPORTED,
)
from src import llm


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — retrieval gate.  Cheap, no LLM call.
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_gate(hits: list[dict]) -> tuple[bool, float]:
    """Did retrieval surface anything plausibly relevant?

    Returns (passed, top_score). `passed` is False when the single best chunk
    is below TAU_RETRIEVE — i.e. nothing in the corpus looks related, so we can
    abstain before spending an LLM token.

    Note (documented finding): this catches true noise ("asdfgh", or questions
    about unrelated domains) but NOT lexical near-misses — e.g. "capital of
    France" scores ~0.44 here because it overlaps the French Open article. Those
    near-misses are the job of SIGNAL 2.
    """
    if not hits:
        return False, 0.0
    top = hits[0]["score"]
    return top >= TAU_RETRIEVE, top


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 2 — groundedness / support check.  One extra LLM call.
# ─────────────────────────────────────────────────────────────────────────────
def support(question: str, answer: str, hits: list[dict]) -> dict:
    """Is `answer` actually backed by `hits`?

    Thin wrapper over llm.support_check that also computes a boolean
    `is_supported` according to the config policy for "partially". Returns
    {"label", "reason", "is_supported"}.
    """
    verdict = llm.support_check(question, answer, hits)
    label = verdict["label"]
    if label == "supported":
        is_supported = True
    elif label == "partially":
        is_supported = not TREAT_PARTIAL_AS_UNSUPPORTED
    else:  # "unsupported"
        is_supported = False
    return {**verdict, "is_supported": is_supported}


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 3 — self-consistency across evidence subsets.  [wired in Stage 5]
# ─────────────────────────────────────────────────────────────────────────────
def disagrees_across_subsets(question: str, hits: list[dict]) -> tuple[bool, float]:
    """Placeholder for Stage 5.

    Will generate the answer from two different evidence subsets and compare
    them via embedding cosine similarity; below TAU_CONSISTENCY => they
    disagree => the evidence is thin/contradictory => abstain.
    """
    raise NotImplementedError("Self-consistency arrives in Stage 5.")
