"""Central config: every tunable in one place.

The whole point of the project is the uncertainty layer, and the uncertainty
layer is governed by the thresholds below. Keeping them here (not scattered
through the code) is what lets evaluate.py treat them as knobs and lets you
tell a precision/recall-tradeoff story in an interview.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "data" / "corpus"      # fetched .txt articles, one per article
INDEX_DIR = ROOT / "data" / "index"        # FAISS index + chunk metadata (built by ingest.py)
EVAL_PATH = ROOT / "data" / "eval.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings & retrieval
# ─────────────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU, strong baseline. The obvious
# default for a lightweight project — upgrading to a bigger model is a one-line
# change here if you ever want to measure the effect.
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_CHARS = 500        # ~100 tokens; small enough that a hit is specific, big enough to carry a fact
CHUNK_OVERLAP = 80       # keep sentences that straddle a boundary retrievable
TOP_K = 5                # passages retrieved per query (first pass)
TOP_K_RETRY = 8          # wider net on SEARCH_AGAIN

# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty thresholds  — the knobs that matter
# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 — retrieval gate. We L2-normalize embeddings and use inner-product
# FAISS, so a "score" is cosine similarity in [-1, 1]. For MiniLM, in-corpus
# question↔chunk pairs typically land ~0.3–0.6; unrelated pairs sit near 0.
# If the single best chunk is below this, nothing relevant was found → abstain
# before spending an LLM token. Tune against your own corpus with evaluate.py.
TAU_RETRIEVE = 0.30

# SIGNAL 3 — self-consistency. Two answers from different evidence subsets are
# "in agreement" if their embeddings' cosine similarity is at least this. Below
# it, the evidence is thin/contradictory → abstain. Only used when enabled below.
TAU_CONSISTENCY = 0.75

# Signal 2 (LLM support check) has no numeric threshold — it returns a label.
# "unsupported" triggers SEARCH_AGAIN/ABSTAIN; "partially" is treated as a pass
# in v1 (flip this if you want to be stricter).
TREAT_PARTIAL_AS_UNSUPPORTED = False

# Toggle for the third signal. Flip to True to enable Stage 5 self-consistency.
ENABLE_SELF_CONSISTENCY = True

# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────
# Backend switch: "ollama" runs locally for free; "anthropic" requires an API key.
# Flip this one string to swap the entire LLM layer — nothing else needs changing.
LLM_BACKEND = "ollama"

# Ollama — local server, OpenAI-compatible endpoint.
# Run `ollama pull gemma3:4b` once, then `ollama serve` (or the desktop app).
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL    = "gemma3:4b"

# Anthropic — only used when LLM_BACKEND = "anthropic". Requires ANTHROPIC_API_KEY in .env.
# Swap for a bigger model by changing this one string.
LLM_MODEL = "claude-haiku-4-5"
LLM_MAX_TOKENS = 1024
# Low temperature everywhere: we want consistent, grounded output and a stable
# judge, not creativity.
LLM_TEMPERATURE = 0.0
