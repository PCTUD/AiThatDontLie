# AiThatDontLie

**A RAG chatbot that knows when to answer, search again, or refuse.**

Most RAG systems hallucinate confidently when they don't know something. This project
explores three cheap, independent uncertainty signals that let a retrieval pipeline say
*"I don't know"* instead of inventing an answer — and measures the tradeoff honestly.

---

## Quick start

```bash
git clone https://github.com/PCTUD/AiThatDontLie.git
cd AiThatDontLie
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                 # add your ANTHROPIC_API_KEY
python -m src.ingest                                 # build the FAISS index (~60s)
streamlit run app.py
```

For a plain-language walkthrough of the UI, see **[USAGE.md](USAGE.md)**.  
For developer / contributor docs, see **[CLAUDE.md](CLAUDE.md)**.

---

## What it does

Ask it a tennis question. It retrieves relevant passages from a ~24-article Wikipedia
corpus, generates a grounded answer with inline citations, and makes one of three decisions:

| Decision | Meaning |
|----------|---------|
| 🟢 **ANSWER** | Evidence supports the answer. |
| 🟡 **SEARCH AGAIN → answered** | Initial retrieval was weak; wider search found support. |
| 🔴 **ABSTAIN** | Evidence doesn't support a confident answer. System refuses. |

The Streamlit UI lets you toggle between the **Abstaining RAG** pipeline and a **Basic RAG
baseline** that always answers — so you can see the difference on the same question.

---

## Architecture

```
question
   │
   ▼
retriever.py ──── FAISS + Sentence Transformers (all-MiniLM-L6-v2)
   │              cosine similarity over 2080 chunks from 24 articles
   ▼
router.py ─────── the decision loop (see below)
   │    │
   │    └── uncertainty.py ── three signals:
   │                          1. retrieval gate   (free)
   │                          2. LLM support check (1 extra call)
   │                          3. self-consistency  (2 extra calls)
   ▼
llm.py ─────────── single Claude API touch-point
                   generate() + support_check(), structured JSON output
```

### The decision loop

```
hits = retriever.search(question, k=5)

SIGNAL 1 — retrieval gate (free):
  if top hit score < TAU_RETRIEVE (0.30):
      → ABSTAIN  [no LLM call spent]

gen = llm.generate(question, hits)

SIGNAL 2 — LLM support check (1 extra call):
  if answer not grounded in passages:
      hits = retriever.search(question, k=8)   ← SEARCH_AGAIN
      gen = llm.generate(question, hits)
      if still not grounded:
          → ABSTAIN

SIGNAL 3 — self-consistency (2 extra calls, optional):
  generate answer from top-half passages and bottom-half passages separately
  if embedding similarity between the two answers < TAU_CONSISTENCY (0.75):
      → ABSTAIN  [evidence is contradictory or thin]

→ ANSWER
```

---

## The three uncertainty signals

### Signal 1 — retrieval gate

Embed the query and chunks with Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim,
L2-normalised). Use FAISS inner product (= cosine similarity) to retrieve the top-K
chunks. If the single best score is below `TAU_RETRIEVE = 0.30`, nothing relevant was
found — abstain before spending any LLM tokens.

**What it catches:** true out-of-domain questions ("who invented the iPhone?" → 0.21).  
**What it misses:** lexical near-misses. "Capital of France?" scores **0.44** because
the word "France" overlaps the French Open article. Signal 1 passes it through — signal 2
catches it.

### Signal 2 — LLM support check

After generating a grounded answer, make a *second, independent* LLM call:

> "Here is a question, an answer, and the passages it was supposed to be based on.
> Is every claim in the answer directly supported by the passages?
> Reply: supported / partially / unsupported."

This is a harder-to-fool judge than asking the model to self-rate confidence, because:
- It's a fresh call — the model never sees its own reasoning chain.
- It sees only the (answer, passages) pair, not the generation prompt.
- It's classifier-style (structured JSON output), low temperature, forced enum.

If the verdict is `unsupported`, the router tries a wider retrieval (SEARCH_AGAIN)
before giving up. `partially` is treated as a pass by default
(`TREAT_PARTIAL_AS_UNSUPPORTED = False` in config).

### Signal 3 — self-consistency

Split the retrieved hits into two halves (top-ranked vs lower-ranked), generate an
independent answer from each, and compare the two answers via embedding cosine similarity.
Below `TAU_CONSISTENCY = 0.75` the answers diverge — the evidence is thin or
contradictory, so abstain.

This runs only when `ENABLE_SELF_CONSISTENCY = True` (the default) and costs two extra
LLM calls. It catches cases where the corpus contains conflicting passages about the same
fact — something signals 1 and 2 can't detect.

---

## Eval harness

```bash
python -m src.evaluate
```

Runs both pipelines over 22 hand-labelled questions in `data/eval.jsonl`:
- 14 **answerable** (in-corpus tennis questions)
- 8 **unanswerable** — including:
  - out-of-corpus in-topic questions (Steffi Graf, Pete Sampras — their articles were
    rate-limited during ingestion; natural "I don't know" cases)
  - adversarial near-misses ("capital of France?", "who won Wimbledon in 2024?")
  - pure off-domain ("who invented the iPhone?")

**Key metric: FP (unsupported answers)** — unanswerable questions where the pipeline
answered anyway. A good abstaining system drives this low without wrongly refusing many
answerable questions (FN).

| Metric | Basic RAG | Abstaining RAG |
|--------|-----------|----------------|
| Correctly answered (TP) | — | — |
| Wrongly refused (FN) | — | — |
| Correctly abstained (TN) | — | — |
| **Unsupported answers (FP) ▼** | — | — |
| Precision | — | — |
| Recall | — | — |

*Run `python -m src.evaluate` to fill this table — it requires `ANTHROPIC_API_KEY`.*

---

## Configuration

All tunables live in `config.py`. The uncertainty thresholds are the interesting knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `TAU_RETRIEVE` | `0.30` | Retrieval gate threshold. Lower → answer more borderline queries. |
| `TAU_CONSISTENCY` | `0.75` | Self-consistency threshold. Lower → let more divergent answers through. |
| `TREAT_PARTIAL_AS_UNSUPPORTED` | `False` | `True` → stricter; partial support triggers SEARCH_AGAIN. |
| `ENABLE_SELF_CONSISTENCY` | `True` | `False` → skip signal 3 (faster, cheaper eval). |
| `TOP_K` / `TOP_K_RETRY` | `5` / `8` | Passages retrieved on first pass / retry. |
| `LLM_MODEL` | `claude-haiku-4-5` | Any Claude model ID — one-line swap. |

---

## Technology stack

| Component | Library |
|-----------|---------|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| Vector search | `faiss-cpu` — `IndexFlatIP` (inner product = cosine on L2-normalised vectors) |
| LLM | `anthropic` — `claude-haiku-4-5`, structured JSON output via `output_config.format` |
| UI | `streamlit` |
| Env | `python-dotenv` |

---

## Project structure

```
AiThatDontLie/
├── config.py           # every tunable in one place
├── app.py              # Streamlit UI
├── requirements.txt
├── .env.example
├── CLAUDE.md           # developer guide
├── USAGE.md            # plain-language user guide
├── data/
│   ├── corpus/         # 24 tennis Wikipedia articles (committed)
│   ├── index/          # FAISS index + chunk metadata (gitignored)
│   └── eval.jsonl      # 22 labelled eval questions
└── src/
    ├── ingest.py       # fetch → chunk → embed → FAISS index
    ├── retriever.py    # search(query, k) → [{text, source, score}]
    ├── llm.py          # generate() + support_check() — Claude API
    ├── uncertainty.py  # the three signals
    ├── router.py       # basic_rag() + abstaining_rag()
    └── evaluate.py     # comparison table over eval.jsonl
```

---

## Key findings

- **Signal 1 alone isn't enough.** A retrieval gate at 0.30 catches true noise but lets
  lexical near-misses through ("capital of France" scores 0.44 via French Open overlap).
  This motivates the layered design — each signal catches a different failure mode.

- **The independent judge matters.** Signal 2 makes a separate LLM call that never sees
  its own generation reasoning. That separation is what makes it harder to fool than
  asking the model to self-rate confidence.

- **SEARCH_AGAIN is cheap insurance.** One wider retrieval before giving up surfaces
  passages the first pass missed; the cost is one extra retrieval + one extra generate
  call, which is often worth it.

- **Self-consistency is expensive but targeted.** Two extra LLM calls per question that
  passes signals 1 and 2. Only fires on thin or contradictory evidence — exactly the
  case the other signals miss. Worth measuring the eval FP reduction per extra call.

---

## Stretch goals (not built)

- Confidence score (0–1) instead of a hard ABSTAIN threshold.
- Query rewriting via LLM on SEARCH_AGAIN (currently just a larger `k`).
- Cross-encoder reranker — measure the effect on abstention quality.
- Highlight the exact sentence in each passage that supports each claim.
- Calibration plot: abstention rate vs. accuracy as `TAU_RETRIEVE` sweeps.
- Adversarial prompts — does the abstaining pipeline resist user pressure ("but are you
  sure it wasn't Nadal?") better than the baseline?

---

## Author

Built by [PCTUD](https://github.com/PCTUD).  
Contributions welcome — open an issue or PR.
