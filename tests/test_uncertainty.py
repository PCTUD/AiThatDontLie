"""Tests for the three uncertainty signals in src/uncertainty.py.

All LLM calls are mocked so these run locally with no API key and no Ollama.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _hit(score: float, text: str = "passage text", source: str = "wiki") -> dict:
    return {"text": text, "source": source, "score": score}


HITS_STRONG = [_hit(0.70), _hit(0.60), _hit(0.50), _hit(0.40)]
HITS_WEAK   = [_hit(0.10), _hit(0.08)]
HITS_EMPTY: list[dict] = []


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1 — retrieval gate
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalGate:
    def test_passes_when_top_score_above_threshold(self) -> None:
        from src.uncertainty import retrieval_gate
        passed, score = retrieval_gate(HITS_STRONG)
        assert passed is True
        assert score == pytest.approx(0.70)

    def test_fires_when_top_score_below_threshold(self) -> None:
        from src.uncertainty import retrieval_gate
        passed, score = retrieval_gate(HITS_WEAK)
        assert passed is False
        assert score == pytest.approx(0.10)

    def test_fires_on_empty_hits(self) -> None:
        from src.uncertainty import retrieval_gate
        passed, score = retrieval_gate(HITS_EMPTY)
        assert passed is False
        assert score == 0.0

    def test_exactly_at_threshold_passes(self) -> None:
        from config import TAU_RETRIEVE
        from src.uncertainty import retrieval_gate
        passed, _ = retrieval_gate([_hit(TAU_RETRIEVE)])
        assert passed is True

    def test_just_below_threshold_fires(self) -> None:
        from config import TAU_RETRIEVE
        from src.uncertainty import retrieval_gate
        passed, _ = retrieval_gate([_hit(TAU_RETRIEVE - 0.001)])
        assert passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2 — support check
# ─────────────────────────────────────────────────────────────────────────────

class TestSupport:
    def _call(self, label: str) -> dict:
        from src.uncertainty import support
        with patch("src.uncertainty.llm") as mock_llm:
            mock_llm.support_check.return_value = {
                "label": label,
                "reason": "mocked reason",
            }
            return support("Q?", "some answer", HITS_STRONG)

    def test_supported_label_is_supported(self) -> None:
        result = self._call("supported")
        assert result["is_supported"] is True
        assert result["label"] == "supported"

    def test_unsupported_label_is_not_supported(self) -> None:
        result = self._call("unsupported")
        assert result["is_supported"] is False

    def test_partially_treated_as_supported_by_default(self) -> None:
        result = self._call("partially")
        assert result["is_supported"] is True

    def test_partially_treated_as_unsupported_when_strict(self) -> None:
        from src.uncertainty import support
        with patch("src.uncertainty.llm") as mock_llm:
            mock_llm.support_check.return_value = {"label": "partially", "reason": ""}
            with patch("src.uncertainty.TREAT_PARTIAL_AS_UNSUPPORTED", True):
                result = support("Q?", "answer", HITS_STRONG)
        assert result["is_supported"] is False

    def test_reason_forwarded(self) -> None:
        result = self._call("supported")
        assert result["reason"] == "mocked reason"


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3 — self-consistency
# ─────────────────────────────────────────────────────────────────────────────

def _unit_vec(dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestSelfConsistency:
    def _run(self, sim: float) -> tuple[bool, float]:
        """Run disagrees_across_subsets with mock LLM + embed model at given similarity."""
        from src.uncertainty import disagrees_across_subsets

        v0 = _unit_vec(seed=0)
        # Build v1 such that dot(v0, v1) ≈ sim by rotating v0 toward an orthogonal vector.
        v_orth = _unit_vec(seed=1)
        v_orth -= v_orth.dot(v0) * v0
        v_orth /= np.linalg.norm(v_orth)
        angle = np.arccos(np.clip(sim, -1.0, 1.0))
        v1 = np.cos(angle) * v0 + np.sin(angle) * v_orth

        mock_model = MagicMock()
        mock_model.encode.return_value = np.stack([v0, v1])

        with patch("src.uncertainty.llm") as mock_llm, \
             patch("src.uncertainty._get_embed_model", return_value=mock_model):
            mock_llm.generate.return_value = {"answer": "some answer", "used": [1]}
            return disagrees_across_subsets("Q?", HITS_STRONG)

    def test_high_similarity_does_not_disagree(self) -> None:
        disagrees, sim = self._run(sim=0.95)
        assert disagrees is False
        assert sim == pytest.approx(0.95, abs=0.01)

    def test_low_similarity_disagrees(self) -> None:
        disagrees, sim = self._run(sim=0.50)
        assert disagrees is True
        assert sim == pytest.approx(0.50, abs=0.01)

    def test_skipped_when_fewer_than_four_hits(self) -> None:
        from src.uncertainty import disagrees_across_subsets
        disagrees, sim = disagrees_across_subsets("Q?", HITS_STRONG[:3])
        assert disagrees is False
        assert sim == 1.0

    def test_skipped_on_empty_hits(self) -> None:
        from src.uncertainty import disagrees_across_subsets
        disagrees, sim = disagrees_across_subsets("Q?", HITS_EMPTY)
        assert disagrees is False
        assert sim == 1.0
