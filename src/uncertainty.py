"""The uncertainty signals — the part that makes this more than plain RAG.

Each signal is a small function returning a verdict the router can act on.
Keeping them separate (not buried in router.py) means each can be unit-tested
and reasoned about on its own, and the eval can attribute an abstention to a
specific signal.

    SIGNAL 1  retrieval_gate(hits)                 -> (passed: bool, top_score)
    SIGNAL 2  support(question, answer, hits)       -> label + reason   (wraps llm.support_check)
    SIGNAL 3  disagrees_across_subsets(...)         -> (disagrees, sim)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL,
    TAU_CONSISTENCY,
    TAU_RETRIEVE,
    TREAT_PARTIAL_AS_UNSUPPORTED,
)
from src import llm

# Lazy singleton — same model already loaded by retriever.py in most runs,
# but we don't want a circular import so we load it here independently.
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


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
# SIGNAL 3 — self-consistency across evidence subsets.
# ─────────────────────────────────────────────────────────────────────────────
def disagrees_across_subsets(question: str, hits: list[dict]) -> tuple[bool, float]:
    """Do two different evidence subsets produce contradictory answers?

    Strategy: split the top-K hits into two overlapping halves — subset A is
    the top half, subset B is the bottom half.  If there are fewer than 4 hits
    we can't form two meaningful subsets, so we skip the check (return False).

    We generate an answer from each subset independently, then compare the two
    answers by embedding cosine similarity.  Below TAU_CONSISTENCY the answers
    diverge, meaning the evidence is thin or internally contradictory — abstain.

    Why this matters: a question like "Who won Wimbledon in 1975?" may have one
    passage that says the correct name and another that names a different finalist.
    Both subsets produce different answers; low similarity flags it.

    Cost: two extra LLM calls, one embed call.  Only runs when
    ENABLE_SELF_CONSISTENCY is True in config.

    Returns (disagrees: bool, similarity: float).
    """
    n = len(hits)
    if n < 4:
        # Not enough hits to form two distinct, non-trivial subsets — skip.
        return False, 1.0

    mid = n // 2
    subset_a = hits[:mid]           # top-ranked passages
    subset_b = hits[mid:]           # lower-ranked passages

    gen_a = llm.generate(question, subset_a)
    gen_b = llm.generate(question, subset_b)

    answer_a = gen_a["answer"]
    answer_b = gen_b["answer"]

    model = _get_embed_model()
    vecs = model.encode([answer_a, answer_b], normalize_embeddings=True).astype(np.float32)
    sim = float(np.dot(vecs[0], vecs[1]))   # cosine similarity (already L2-normalized)

    disagrees = sim < TAU_CONSISTENCY
    return disagrees, sim
