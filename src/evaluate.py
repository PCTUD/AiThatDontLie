"""Eval harness: run both pipelines over eval.jsonl and print a comparison table.

Usage:
    python -m src.evaluate

Requires ANTHROPIC_API_KEY in .env (for abstaining_rag's LLM support check).

The key metric is **unsupported answers** — cases where a pipeline returned
ANSWER/SEARCH_AGAIN but the question was tagged `unanswerable`. A good
abstaining system drives this number down without refusing many answerable ones.

Exit code is 0; the table goes to stdout. Pipe to `| tee results.txt` to keep.
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Ensure project root is on path when run via `python -m src.evaluate`
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EVAL_PATH
from src import router
from src.router import Result

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_questions() -> list[dict]:
    with open(EVAL_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_pipeline(name: str, fn: Callable[[str], Result], questions: list[dict]) -> list[dict]:
    """Run `fn` over every question; return a list of result dicts."""
    rows = []
    for q in questions:
        print(f"  [{name}] Q{q['id']:02d}: {q['question'][:60]}...", flush=True)
        t0 = time.monotonic()
        try:
            result = fn(q["question"])
            elapsed = time.monotonic() - t0
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "note": q.get("note", ""),
                "decision": result.decision,
                "answer": result.answer,
                "top_score": result.signals.get("top_score"),
                "support": result.signals.get("support"),
                "consistency_sim": result.signals.get("consistency_sim"),
                "elapsed": round(elapsed, 2),
                "error": None,
            })
        except Exception as exc:
            elapsed = time.monotonic() - t0
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "note": q.get("note", ""),
                "decision": "ERROR",
                "answer": "",
                "top_score": None,
                "support": None,
                "consistency_sim": None,
                "elapsed": round(elapsed, 2),
                "error": str(exc),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] == "answerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]

    # True positives: answerable questions that got ANSWER or SEARCH_AGAIN
    tp = sum(1 for r in answerable if r["decision"] in {"ANSWER", "SEARCH_AGAIN"})
    # False negatives: answerable questions wrongly abstained
    fn = sum(1 for r in answerable if r["decision"] == "ABSTAIN")
    # True negatives: unanswerable questions correctly abstained
    tn = sum(1 for r in unanswerable if r["decision"] == "ABSTAIN")
    # False positives: unanswerable questions that got an answer (unsupported answers)
    fp = sum(1 for r in unanswerable if r["decision"] in {"ANSWER", "SEARCH_AGAIN"})
    errors = sum(1 for r in rows if r["decision"] == "ERROR")

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    return {
        "total": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "tp": tp,   # correctly answered
        "fn": fn,   # wrongly refused (answerable → ABSTAIN)
        "tn": tn,   # correctly abstained
        "fp": fp,   # unsupported answers (the number we want low)
        "errors": errors,
        "precision": precision,  # of all answers given, fraction on answerable questions
        "recall": recall,        # of all answerable questions, fraction correctly answered
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printing
# ─────────────────────────────────────────────────────────────────────────────

DECISION_ICON = {"ANSWER": "🟢", "SEARCH_AGAIN": "🟡", "ABSTAIN": "🔴", "ERROR": "💥"}


def _decision_str(decision: str) -> str:
    return f"{DECISION_ICON.get(decision, '?')} {decision}"


def print_detail_table(name: str, rows: list[dict]) -> None:
    print(f"\n{'═' * 110}")
    print(f"  {name} — per-question results")
    print(f"{'═' * 110}")
    print(f"  {'ID':>2}  {'Type':>11}  {'Decision':>22}  {'Score':>5}  {'Support':>10}  {'ConsSim':>7}  Question")
    print(f"  {'─'*2}  {'─'*11}  {'─'*22}  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*50}")
    for r in rows:
        score_s = f"{r['top_score']:.2f}" if r["top_score"] is not None else "  — "
        sup_s = r["support"] or "—"
        sim = r.get("consistency_sim")
        sim_s = f"{sim:.2f}" if sim is not None else "  —  "
        decision_s = _decision_str(r["decision"])
        q = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        print(f"  {r['id']:>2}  {r['type']:>11}  {decision_s:>22}  {score_s:>5}  {sup_s:>10}  {sim_s:>7}  {q}")


def print_summary_table(basic_m: dict, abstain_m: dict) -> None:
    print(f"\n{'═' * 60}")
    print("  SUMMARY COMPARISON")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<30}  {'Basic RAG':>10}  {'Abstaining':>10}")
    print(f"  {'─'*30}  {'─'*10}  {'─'*10}")

    rows = [
        ("Total questions", basic_m["total"], abstain_m["total"]),
        ("  — answerable",  basic_m["answerable"], abstain_m["answerable"]),
        ("  — unanswerable", basic_m["unanswerable"], abstain_m["unanswerable"]),
        ("", None, None),
        ("✅ Correctly answered (TP)",    basic_m["tp"],  abstain_m["tp"]),
        ("❌ Wrongly refused (FN)",        basic_m["fn"],  abstain_m["fn"]),
        ("✅ Correctly abstained (TN)",    basic_m["tn"],  abstain_m["tn"]),
        ("🚨 Unsupported answers (FP) ▼", basic_m["fp"],  abstain_m["fp"]),
        ("💥 Errors",                      basic_m["errors"], abstain_m["errors"]),
        ("", None, None),
        ("Precision (answers on-target)", f"{basic_m['precision']:.0%}", f"{abstain_m['precision']:.0%}"),
        ("Recall (answerable covered)",   f"{basic_m['recall']:.0%}",    f"{abstain_m['recall']:.0%}"),
    ]

    for label, b, a in rows:
        if label == "":
            print()
            continue
        b_s = str(b) if b is not None else ""
        a_s = str(a) if a is not None else ""
        # Highlight the FP row — it's the one that matters
        marker = " ◄" if "FP" in label else ""
        print(f"  {label:<30}  {b_s:>10}  {a_s:>10}{marker}")

    print("\n  ▼  Lower unsupported answers = better abstaining pipeline.")
    print("  ▼  FN (wrongly refused) is the cost — we want it low too.")
    print(f"{'═' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    questions = load_questions()
    print(f"\nLoaded {len(questions)} eval questions from {EVAL_PATH}\n")

    print("Running Basic RAG pipeline...")
    basic_rows = run_pipeline("Basic RAG", router.basic_rag, questions)

    print("\nRunning Abstaining RAG pipeline...")
    abstain_rows = run_pipeline("Abstaining RAG", router.abstaining_rag, questions)

    basic_m = compute_metrics(basic_rows)
    abstain_m = compute_metrics(abstain_rows)

    print_detail_table("Basic RAG", basic_rows)
    print_detail_table("Abstaining RAG", abstain_rows)
    print_summary_table(basic_m, abstain_m)

    print(f"\nDone. Total wall time: basic ~{sum(r['elapsed'] for r in basic_rows):.1f}s, "
          f"abstaining ~{sum(r['elapsed'] for r in abstain_rows):.1f}s\n")


if __name__ == "__main__":
    main()
