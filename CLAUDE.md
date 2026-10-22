# CLAUDE.md — Developer Guide

This file tells Claude Code (and human developers) how this project is structured,
what the conventions are, and how to extend it without breaking things.

## Project in one sentence

A RAG chatbot over a tennis Wikipedia corpus that can ANSWER, SEARCH_AGAIN, or
ABSTAIN — with three independent uncertainty signals to decide which.

## Repo layout

```
AiThatDontLie/
├── config.py            # Every tunable in one place. Touch this first.
├── app.py               # Streamlit UI. Thin wrapper over router.py.
├── requirements.txt
├── .env.example         # Copy to .env, add ANTHROPIC_API_KEY
├── data/
│   ├── corpus/          # 24 tennis .txt articles (committed, self-contained)
│   ├── index/           # faiss.index + chunks.json  (gitignored — regenerate)
│   └── eval.jsonl       # 22 tagged questions: answerable / unanswerable
└── src/
    ├── ingest.py        # fetch Wikipedia → chunk → embed → FAISS index
    ├── retriever.py     # load index; search(query, k) → [{text, source, score}]
    ├── llm.py           # ONE LLM touch-point: generate(), support_check() — Ollama or Anthropic
    ├── uncertainty.py   # the three signals as pure functions
    ├── router.py        # decision loop: basic_rag() vs abstaining_rag()
    └── evaluate.py      # run both pipelines over eval.jsonl, print comparison table
```

## Setup from scratch

### Using Ollama (free, no API key)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull gemma3:4b              # one-time download (~3GB); needs Ollama installed
python -m src.ingest               # build data/index/ (takes ~60s first time)
python -m src.llm                  # smoke test — should print "OK"
streamlit run app.py
```

`LLM_BACKEND = "ollama"` is already the default in `config.py`. No `.env` needed.

### Using Anthropic Claude API instead

```bash
cp .env.example .env               # add your ANTHROPIC_API_KEY
# in config.py: set LLM_BACKEND = "anthropic"
python -m src.llm                  # smoke test
streamlit run app.py
```

The corpus (`data/corpus/`) is already committed — you only need to re-run
`python -m src.ingest` if you add articles or change `CHUNK_CHARS`.

## The three uncertainty signals

All three live in `src/uncertainty.py`. The router in `src/router.py` calls them
in order — each can trigger ABSTAIN independently.

| # | Signal | Cost | What it catches |
|---|--------|------|-----------------|
| 1 | **Retrieval gate** — `retrieval_gate(hits)` | Free (no LLM) | Questions with no relevant passages in the corpus (e.g. "Who invented the iPhone?"). Fires before any LLM call. |
| 2 | **Support check** — `support(question, answer, hits)` | 1 extra LLM call | Answers that aren't backed by the retrieved passages (e.g. "capital of France" slips past signal 1 via French Open overlap; signal 2 kills it). |
| 3 | **Self-consistency** — `disagrees_across_subsets(question, hits)` | 2 extra LLM calls | Contradictory evidence — splits hits into two halves, generates independently, ABSTAINs if embedding sim < `TAU_CONSISTENCY`. Gated by `ENABLE_SELF_CONSISTENCY`. |

**The key design insight:** signal 1 is free and eliminates true noise; signal 2
is a second LLM call that acts as a harder-to-fool judge (it never sees its own
reasoning — only the answer and the passages); signal 3 is expensive but catches
thin/contradictory evidence that the other two miss.

## The decision loop (`router.abstaining_rag`)

```
hits = retriever.search(question, k=TOP_K)

SIGNAL 1: if top hit score < TAU_RETRIEVE → ABSTAIN (no LLM spent)

gen = llm.generate(question, hits)
SIGNAL 2: support_check(question, gen["answer"], hits)
  if unsupported:
      hits = retriever.search(question, k=TOP_K_RETRY)  ← SEARCH_AGAIN
      gen = llm.generate(question, hits)
      support_check again
      if still unsupported → ABSTAIN

SIGNAL 3 (if ENABLE_SELF_CONSISTENCY):
  disagrees_across_subsets(question, hits)
  if disagrees → ABSTAIN

→ ANSWER (or SEARCH_AGAIN if we retried)
```

## Config knobs (config.py)

| Knob | Default | Effect |
|------|---------|--------|
| `LLM_BACKEND` | `"ollama"` | `"ollama"` = local free model; `"anthropic"` = Claude API (needs key) |
| `OLLAMA_MODEL` | `"gemma3:4b"` | Ollama model tag — must match what you've pulled |
| `OLLAMA_BASE_URL` | `"http://localhost:11434/v1"` | Ollama server address |
| `LLM_MODEL` | `"claude-haiku-4-5"` | Anthropic model — only used when `LLM_BACKEND = "anthropic"` |
| `TAU_RETRIEVE` | 0.30 | Lower → answer more, abstain less on ambiguous queries |
| `TAU_CONSISTENCY` | 0.75 | Lower → let more divergent answers through |
| `TREAT_PARTIAL_AS_UNSUPPORTED` | False | True → stricter; partially supported answers trigger SEARCH_AGAIN |
| `ENABLE_SELF_CONSISTENCY` | True | False → skip signal 3 (faster, cheaper) |
| `TOP_K` / `TOP_K_RETRY` | 5 / 8 | More passages = more context but slower and noisier |

## LLM backend (`src/llm.py`)

All LLM calls live here. The project uses two backends dispatched by `LLM_BACKEND`:

**Ollama path** — `openai.OpenAI` pointed at `localhost:11434/v1` (Ollama's
OpenAI-compatible endpoint). Uses `response_format={"type": "json_object"}` for JSON
output, plus explicit JSON format instructions in the system prompt. A `_safe_parse()`
helper gracefully handles any field-presence failures since Ollama doesn't enforce schemas.

**Anthropic path** — `anthropic.Anthropic` with `output_config.format` (JSON schema
enforcement, server-side). Unchanged from the original implementation.

Both paths expose identical public functions — `generate()`, `support_check()`,
`smoke_test()` — so `router.py` and `uncertainty.py` need zero changes when you
switch backends. The lazy client singleton means signal-1-only paths never start Ollama.

## Adding a new signal

1. Write a function in `src/uncertainty.py` that returns `(fires: bool, value: float)`.
2. Add a threshold to `config.py` and a toggle if it's optional.
3. Add a call in `router.abstaining_rag()` in the right place in the chain.
4. Capture the signal value into `signals` dict so the eval and UI can surface it.
5. Add a question to `data/eval.jsonl` that exercises the new signal.

## Running the eval

```bash
python -m src.evaluate
```

With `LLM_BACKEND = "ollama"`, no API key needed — runs fully locally. With
`LLM_BACKEND = "anthropic"`, requires `ANTHROPIC_API_KEY` in `.env`. Runs both
pipelines over all 22 eval questions and prints a comparison table — the key
metric is **FP (unsupported answers)**: how many unanswerable questions the
pipeline answered anyway.

## Extending the corpus

```bash
# Edit src/ingest.py — add article titles to the ARTICLES list, then:
python -m src.ingest
```

After ingestion, add matching questions to `data/eval.jsonl` so the eval stays
representative. Re-run `python -m src.evaluate` to see the effect.

## What deliberately doesn't exist here

- No database, no persistent sessions — every query is stateless.
- No streaming — Haiku is fast enough that buffering the full response is fine.
- No retry logic in `router.py` beyond the one SEARCH_AGAIN pass — adding more
  makes the abstention story muddier. If you need deeper retry, add a second
  `TOP_K_RETRY2` pass and document the precision/recall tradeoff.
- No query rewriting on SEARCH_AGAIN — the retry just uses a larger `k`. Query
  rewriting is a good stretch goal (see README).
