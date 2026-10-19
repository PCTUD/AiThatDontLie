"""Stage 1 — build the corpus and the FAISS index.

Pipeline:  fetch Wikipedia articles  ->  chunk  ->  embed  ->  FAISS index

Run once:   python -m src.ingest
Re-run any time; it overwrites data/corpus/ and data/index/.

The theme is tennis — chosen because it yields clean *answerable* questions
("Who won the most Grand Slam titles?") and natural *unanswerable* ones that are
on-topic but outside a 30-article corpus ("What is the exact ATP ranking
formula?"). That answerable/unanswerable split is what the abstention eval needs.
"""

import json
import re
import sys
import time
import urllib.parse
import urllib.request

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_CHARS,
    CHUNK_OVERLAP,
    CORPUS_DIR,
    EMBED_MODEL,
    INDEX_DIR,
)

# Wikipedia article titles for the corpus. ~30 tennis articles: players, majors,
# equipment, rules, surfaces. Edit this list to re-theme the whole project.
ARTICLES = [
    "Tennis", "Grand Slam (tennis)", "Wimbledon Championships",
    "US Open (tennis)", "French Open", "Australian Open",
    "Roger Federer", "Rafael Nadal", "Novak Djokovic", "Serena Williams",
    "Steffi Graf", "Pete Sampras", "Andre Agassi", "Martina Navratilova",
    "Billie Jean King", "Tennis court", "Clay court", "Grass court",
    "Hard court", "Tennis racket", "Tennis ball", "Tiebreak",
    "ATP Tour", "WTA Tour", "Davis Cup", "Fed Cup",
    "Tennis scoring system", "Serve (tennis)", "Volley (tennis)",
    "Grand Slam (tennis) champions in Open Era",
]

# Wikipedia REST "plain text extract" endpoint — no API key, no extra dependency.
WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-abstain-demo/0.1 (educational project)"


def fetch_article(title: str, max_retries: int = 4) -> str | None:
    """Return the plain-text body of a Wikipedia article, or None on failure.

    Retries on HTTP 429 (rate limit) with exponential backoff, since fetching
    ~30 articles in a loop reliably trips Wikipedia's rate limiter otherwise.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",   # strip HTML, give us plain text
        "redirects": "1",     # follow "Fed Cup" -> "Billie Jean King Cup" etc.
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)   # 2s, 4s, 8s
                print(f"  . rate-limited on {title!r}, waiting {wait}s...",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"  ! fetch failed for {title!r}: {e}", file=sys.stderr)
            return None
        except Exception as e:  # noqa: BLE001 — one-shot script; log and skip
            print(f"  ! fetch failed for {title!r}: {e}", file=sys.stderr)
            return None
    else:
        return None

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract")
        if text and len(text) > 200:   # skip stubs / missing pages
            return text
    return None


def clean(text: str) -> str:
    """Drop Wikipedia section headers (== Foo ==) and collapse whitespace."""
    text = re.sub(r"==+\s*[^=]+?\s*==+", " ", text)   # section headings
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk(text: str) -> list[str]:
    """Fixed-size character chunks with overlap.

    Simplest thing that works. Character windows (not sentence-aware) keep the
    code tiny; the overlap means a fact split across a boundary stays findable.
    Upgrading to sentence-aware chunking is an obvious stretch goal.
    """
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = start + CHUNK_CHARS
        piece = text[start:end].strip()
        if len(piece) > 50:        # drop tiny tail fragments
            chunks.append(piece)
        start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fetch + save raw articles, and build the flat list of (chunk, source).
    all_chunks: list[str] = []
    sources: list[str] = []
    print(f"Fetching {len(ARTICLES)} articles...")
    for title in ARTICLES:
        body = fetch_article(title)
        if not body:
            continue
        body = clean(body)
        # Save the raw article so the repo is self-contained and inspectable.
        safe_name = re.sub(r"[^\w\-]+", "_", title) + ".txt"
        (CORPUS_DIR / safe_name).write_text(body, encoding="utf-8")

        pieces = chunk(body)
        all_chunks.extend(pieces)
        sources.extend([title] * len(pieces))
        print(f"  + {title}: {len(pieces)} chunks")
        time.sleep(0.5)   # be polite to the API — lower values trip the 429 limiter

    if not all_chunks:
        sys.exit("No articles fetched — check your network connection.")

    print(f"\n{len(all_chunks)} chunks from {len(set(sources))} articles.")

    # 2. Embed. Normalize so inner-product == cosine similarity (see config note).
    print(f"Embedding with {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    vecs = model.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # 3. Build a FAISS inner-product index (cosine, since vectors are normalized).
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # 4. Save chunk text + source side-by-side, aligned to index row order.
    meta = [{"text": t, "source": s} for t, s in zip(all_chunks, sources)]
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nIndex built: {index.ntotal} vectors, dim {vecs.shape[1]}.")
    print(f"Wrote {INDEX_DIR}/faiss.index and chunks.json")


if __name__ == "__main__":
    main()
