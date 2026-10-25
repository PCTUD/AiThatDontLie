"""The one place the LLM is touched.

Supports two backends, switchable via LLM_BACKEND in config.py:

    "ollama"     — local Ollama server (free, no API key).
                   Requires: `ollama pull gemma3:4b` then `ollama serve`.
                   Uses Ollama's OpenAI-compatible /v1 endpoint.

    "anthropic"  — Claude API (requires ANTHROPIC_API_KEY in .env).

Public API is identical regardless of backend:
    generate(question, hits)          -> {"answer": str, "used": [int]}
    support_check(question, answer, hits) -> {"label": str, "reason": str}
    smoke_test()                      -> "OK"

Everything that calls an LLM goes through here — router.py and uncertainty.py
never import anthropic or openai directly. Swapping backends is a one-line
change in config.py.
"""

import json
import os
from typing import Any, cast

from dotenv import load_dotenv

from config import (
    LLM_BACKEND,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Client singletons — lazy, one per backend
# ─────────────────────────────────────────────────────────────────────────────
_anthropic_client = None
_ollama_client = None


def _get_client() -> Any:
    global _anthropic_client, _ollama_client
    if LLM_BACKEND == "anthropic":
        if _anthropic_client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add "
                    "your key (get one at console.anthropic.com), or set "
                    "LLM_BACKEND = 'ollama' in config.py to use a local model."
                )
            from anthropic import Anthropic
            _anthropic_client = Anthropic()
        return _anthropic_client
    else:
        if _ollama_client is None:
            from openai import OpenAI
            _ollama_client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama",   # required by the SDK but ignored by Ollama
            )
        return _ollama_client


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_passages(hits: list[dict]) -> str:
    """Render retrieved hits as a numbered list the model can cite as [1], [2]..."""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] (source: {h['source']}) {h['text']}")
    return "\n\n".join(lines)


def _safe_parse(text: str, required: list[str], fallback: dict) -> dict:
    """Parse JSON from `text`, returning `fallback` if it's malformed or missing keys.

    Anthropic's output_config enforces schema server-side; Ollama's json_object
    mode only guarantees syntactically valid JSON — this catches field-presence
    failures and parse errors so the pipeline degrades gracefully rather than crashing.
    """
    try:
        data = json.loads(text)
        if all(k in data for k in required):
            return cast(dict[str, Any], data)
    except (json.JSONDecodeError, TypeError):
        pass
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

GENERATE_SYSTEM = """You answer questions using ONLY the numbered passages provided.

Rules:
- Use only facts stated in the passages. Do not add outside knowledge.
- Cite the passage number(s) supporting each claim, inline, like [1] or [2][3].
- If the passages do not contain the answer, say so plainly rather than guessing.
- Be concise: two or three sentences is usually enough."""

_JSON_HINT_GENERATE = """

You MUST respond with exactly this JSON structure and nothing else:
{"answer": "your answer here", "used": [1, 2]}
The "used" array must contain the passage numbers (integers) you cited."""

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

_JSON_HINT_SUPPORT = """

You MUST respond with exactly this JSON and nothing else:
{"label": "supported", "reason": "one sentence explanation"}
The label field MUST be exactly one of: supported, partially, unsupported"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — grounded generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_anthropic(question: str, hits: list[dict]) -> dict:
    passages = _format_passages(hits)
    user = (
        f"Passages:\n{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the rules you were given. Then report which "
        "passage numbers you actually used."
    )
    client = _get_client()
    resp = client.messages.create(
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
                        "used": {"type": "array", "items": {"type": "integer"}},
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


def _generate_ollama(question: str, hits: list[dict]) -> dict:
    passages = _format_passages(hits)
    user = (
        f"Passages:\n{passages}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the rules you were given. Then report which "
        "passage numbers you actually used."
    )
    client = _get_client()
    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GENERATE_SYSTEM + _JSON_HINT_GENERATE},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content
    data = _safe_parse(text, ["answer", "used"], {"answer": text, "used": []})
    return {"answer": data["answer"], "used": data.get("used", [])}


def generate(question: str, hits: list[dict]) -> dict:
    """Produce a grounded answer with inline citations.

    Returns {"answer": str, "used": [passage numbers the model cited]}.
    """
    if LLM_BACKEND == "ollama":
        return _generate_ollama(question, hits)
    return _generate_anthropic(question, hits)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — groundedness / support check  (SIGNAL 2)
# ─────────────────────────────────────────────────────────────────────────────

def _support_check_anthropic(question: str, answer: str, hits: list[dict]) -> dict:
    passages = _format_passages(hits)
    user = (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"PASSAGES:\n{passages}\n\n"
        "Classify the answer's support, with a one-sentence reason."
    )
    client = _get_client()
    resp = client.messages.create(
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
    return cast(dict[str, Any], json.loads(text))


def _support_check_ollama(question: str, answer: str, hits: list[dict]) -> dict:
    passages = _format_passages(hits)
    user = (
        f"QUESTION: {question}\n\n"
        f"ANSWER: {answer}\n\n"
        f"PASSAGES:\n{passages}\n\n"
        "Classify the answer's support, with a one-sentence reason."
    )
    client = _get_client()
    resp = client.chat.completions.create(
        model=OLLAMA_MODEL,
        max_tokens=256,
        temperature=LLM_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SUPPORT_SYSTEM + _JSON_HINT_SUPPORT},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content
    data = _safe_parse(
        text,
        ["label", "reason"],
        {"label": "unsupported", "reason": "parse error — could not classify support"},
    )
    # Normalise: model may return a label outside the enum
    label = data.get("label", "unsupported").lower().strip()
    if label not in {"supported", "partially", "unsupported"}:
        label = "unsupported"
    return {"label": label, "reason": data.get("reason", "")}


def support_check(question: str, answer: str, hits: list[dict]) -> dict:
    """SIGNAL 2: is `answer` actually grounded in `hits`?

    Returns {"label": "supported"|"partially"|"unsupported", "reason": str}.
    Uses a separate LLM call that never sees the generation reasoning — harder
    to fool than asking the model to self-rate confidence.
    """
    if LLM_BACKEND == "ollama":
        return _support_check_ollama(question, answer, hits)
    return _support_check_anthropic(question, answer, hits)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test() -> str:
    """One trivial call to confirm the backend and model are reachable."""
    client = _get_client()
    if LLM_BACKEND == "ollama":
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        return str(resp.choices[0].message.content).strip()
    else:
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=32,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        return str(next(b.text for b in resp.content if b.type == "text")).strip()


if __name__ == "__main__":
    backend = LLM_BACKEND
    model = OLLAMA_MODEL if backend == "ollama" else LLM_MODEL
    print(f"Backend: {backend}  Model: {model}")
    result = smoke_test()
    print(f"Smoke test: {result}")
