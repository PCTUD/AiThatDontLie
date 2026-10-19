"""The one place the Claude API is touched.

Everything that calls an LLM goes through here, so swapping Haiku for a bigger
model, Ollama, or an OpenAI-compatible endpoint is a change to this file only —
the router and uncertainty code never import `anthropic` directly.

Functions:
    generate(question, hits)        -> grounded answer + which passages it used   [Stage 2]
    support_check(question, answer, hits) -> "supported" | "partially" | "unsupported"  [Stage 3]
    smoke_test()                    -> one trivial call, to verify the key works
"""

import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

load_dotenv()   # read ANTHROPIC_API_KEY from .env if present

# Lazy singleton: don't construct the client (or require a key) until first use,
# so `import llm` is cheap and the retriever-only CLI never needs a key.
_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add "
                "your key (get one at console.anthropic.com)."
            )
        _client = Anthropic()   # reads ANTHROPIC_API_KEY from the environment
    return _client


def _format_passages(hits: list[dict]) -> str:
    """Render retrieved hits as a numbered list the model can cite as [1], [2]..."""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] (source: {h['source']}) {h['text']}")
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — grounded generation
# ─────────────────────────────────────────────────────────────────────────────
GENERATE_SYSTEM = """You answer questions using ONLY the numbered passages provided.

Rules:
- Use only facts stated in the passages. Do not add outside knowledge.
- Cite the passage number(s) supporting each claim, inline, like [1] or [2][3].
- If the passages do not contain the answer, say so plainly rather than guessing.
- Be concise: two or three sentences is usually enough."""


def generate(question: str, hits: list[dict]) -> dict:
    """Produce a grounded answer with inline citations.

    Returns {"answer": str, "used": [passage numbers the model cited]}.
    `used` is parsed from the structured output and lets the support check and
    the UI show exactly which passages were relied on.
    """
    passages = _format_passages(hits)
    user = (
        f"Passages:\n{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the rules you were given. Then report which "
        "passage numbers you actually used."
    )

    # Structured output guarantees we get back the answer AND the list of cited
    # passage numbers without regex-scraping the prose. (output_config.format is
    # the current API; the deprecated top-level output_format is avoided.)
    resp = _get_client().messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        system=GENERATE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "used": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["answer", "used"],
                    "additionalProperties": False,
                },
            }
        },
    )
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return {"answer": data["answer"], "used": data.get("used", [])}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — groundedness / support check  (SIGNAL 2)
# ─────────────────────────────────────────────────────────────────────────────
SUPPORT_SYSTEM = """You are a strict fact-checker.

You are given a QUESTION, an ANSWER, and the PASSAGES the answer was supposed to
be based on. Decide whether every factual claim in the ANSWER is directly
supported by the PASSAGES.

- "supported": every claim is backed by the passages.
- "partially": the main claim is backed but some detail is not, or only part of
  the question is answered from the passages.
- "unsupported": the answer makes claims the passages do not back, OR the answer
  itself says the passages don't contain the answer.

Judge ONLY against the passages, not your own knowledge."""


def support_check(question: str, answer: str, hits: list[dict]) -> dict:
    """SIGNAL 2: is `answer` actually grounded in `hits`?

    A fresh call that sees only (question, answer, passages) — never the model's
    own reasoning — so it's a harder-to-fool judge than self-confidence. Returns
    {"label": "supported"|"partially"|"unsupported", "reason": str}.
    """
    passages = _format_passages(hits)
    user = (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"PASSAGES:\n{passages}\n\n"
        "Classify the answer's support, with a one-sentence reason."
    )
    resp = _get_client().messages.create(
        model=LLM_MODEL,
        max_tokens=256,
        temperature=LLM_TEMPERATURE,
        system=SUPPORT_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["supported", "partially", "unsupported"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "reason"],
                    "additionalProperties": False,
                },
            }
        },
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def smoke_test() -> str:
    """One trivial call to confirm the key and model work."""
    resp = _get_client().messages.create(
        model=LLM_MODEL,
        max_tokens=32,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    return next(b.text for b in resp.content if b.type == "text").strip()


if __name__ == "__main__":
    print(smoke_test())
