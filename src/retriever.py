"""Stage 1 — load the FAISS index and search it.

search(query, k) -> list of {text, source, score}, best first.
score is cosine similarity in [-1, 1] (vectors are L2-normalized, index is IP).

CLI for the Stage 1 checkpoint:
    python -m src.retriever "who has won the most grand slams"
"""

import json
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, INDEX_DIR, TOP_K

# Module-level singletons: load the model + index once, reuse across queries.
# Streamlit reruns the script on every interaction, so lazy-loading here (rather
# than at import) keeps the first paint fast and avoids reloading per keystroke.
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[dict] | None = None


def _load() -> None:
    global _model, _index, _chunks
    if _index is not None:
        return
    index_path = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.json"
    if not index_path.exists() or not chunks_path.exists():
        sys.exit(
            "Index not found. Build it first:\n    python -m src.ingest"
        )
    _model = SentenceTransformer(EMBED_MODEL)
    _index = faiss.read_index(str(index_path))
    _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))


def search(query: str, k: int = TOP_K) -> list[dict]:
    """Return the k most similar chunks to `query`, best first."""
    _load()
    assert _model is not None and _index is not None and _chunks is not None

    qv = _model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, idxs = _index.search(qv, k)

    hits = []
    for score, i in zip(scores[0], idxs[0]):
        if i < 0:          # FAISS pads with -1 when fewer than k results exist
            continue
        hit = dict(_chunks[i])          # {text, source}
        hit["score"] = float(score)     # cosine similarity
        hits.append(hit)
    return hits


def _cli() -> None:
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m src.retriever "your query here"')
    query = " ".join(sys.argv[1:])
    hits = search(query)
    print(f"\nQuery: {query}\n" + "─" * 60)
    for rank, h in enumerate(hits, 1):
        print(f"[{rank}] score={h['score']:.3f}  source={h['source']}")
        print(f"    {h['text'][:200]}...\n")


if __name__ == "__main__":
    _cli()
