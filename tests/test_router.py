"""Tests for router.py — basic_rag and abstaining_rag decision logic.

The retriever and LLM are fully mocked so no index or API key is needed.
"""

from unittest.mock import patch

import pytest

from src.router import Result, abstaining_rag, basic_rag

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hit(score: float) -> dict:
    return {"text": "passage", "source": "wiki", "score": score}


STRONG_HITS = [_hit(0.70), _hit(0.60), _hit(0.50), _hit(0.40), _hit(0.35)]
WEAK_HITS   = [_hit(0.10), _hit(0.08)]

GEN_OK      = {"answer": "Some answer.", "used": [1]}
SUP_OK      = {"label": "supported",   "reason": "backed by passage 1", "is_supported": True}
SUP_BAD     = {"label": "unsupported", "reason": "not in passages",     "is_supported": False}


# ─────────────────────────────────────────────────────────────────────────────
# basic_rag — always answers
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicRag:
    def test_always_answers(self) -> None:
        with patch("src.router.retriever") as r, patch("src.router.llm") as mock_llm:
            r.search.return_value = STRONG_HITS
            mock_llm.generate.return_value = GEN_OK
            result = basic_rag("Who won Wimbledon?")
        assert result.decision == "ANSWER"
        assert result.answer == "Some answer."

    def test_returns_result_type(self) -> None:
        with patch("src.router.retriever") as r, patch("src.router.llm") as mock_llm:
            r.search.return_value = STRONG_HITS
            mock_llm.generate.return_value = GEN_OK
            result = basic_rag("Q?")
        assert isinstance(result, Result)

    def test_empty_index_returns_answer(self) -> None:
        with patch("src.router.retriever") as r:
            r.search.return_value = []
            result = basic_rag("Q?")
        assert result.decision == "ANSWER"

    def test_top_score_in_signals(self) -> None:
        with patch("src.router.retriever") as r, patch("src.router.llm") as mock_llm:
            r.search.return_value = STRONG_HITS
            mock_llm.generate.return_value = GEN_OK
            result = basic_rag("Q?")
        assert result.signals["top_score"] == pytest.approx(0.70)


# ─────────────────────────────────────────────────────────────────────────────
# abstaining_rag — signal 1: retrieval gate
# ─────────────────────────────────────────────────────────────────────────────

class TestAbstainingRagSignal1:
    def test_abstains_on_weak_retrieval(self) -> None:
        with patch("src.router.retriever") as r, patch("src.router.llm") as mock_llm:
            r.search.return_value = WEAK_HITS
            result = abstaining_rag("Who invented the iPhone?")
            mock_llm.generate.assert_not_called()
        assert result.decision == "ABSTAIN"

    def test_abstain_reason_mentions_retrieval_gate(self) -> None:
        with patch("src.router.retriever") as r:
            r.search.return_value = WEAK_HITS
            result = abstaining_rag("Q?")
        assert "Retrieval gate" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# abstaining_rag — signal 2: support check
# ─────────────────────────────────────────────────────────────────────────────

class TestAbstainingRagSignal2:
    def _run(self, sup1: dict, sup2: dict | None = None) -> Result:
        """Run abstaining_rag with signal 1 passing and controllable support verdicts."""
        call_count = 0

        def support_side_effect(*_args: object, **_kwargs: object) -> dict:
            nonlocal call_count
            call_count += 1
            return sup1 if call_count == 1 else (sup2 or sup1)

        with patch("src.router.retriever") as r, \
             patch("src.router.llm") as mock_llm, \
             patch("src.router.uncertainty") as u, \
             patch("src.router.ENABLE_SELF_CONSISTENCY", False):
            r.search.return_value = STRONG_HITS
            mock_llm.generate.return_value = GEN_OK
            u.retrieval_gate.return_value = (True, 0.70)
            u.support.side_effect = support_side_effect
            return abstaining_rag("Q?")

    def test_answers_when_supported(self) -> None:
        result = self._run(SUP_OK)
        assert result.decision == "ANSWER"

    def test_abstains_when_both_support_checks_fail(self) -> None:
        result = self._run(SUP_BAD, SUP_BAD)
        assert result.decision == "ABSTAIN"

    def test_search_again_then_answers_when_retry_passes(self) -> None:
        result = self._run(SUP_BAD, SUP_OK)
        assert result.decision == "SEARCH_AGAIN"

    def test_support_signal_in_signals_dict(self) -> None:
        result = self._run(SUP_OK)
        assert "support" in result.signals


# ─────────────────────────────────────────────────────────────────────────────
# abstaining_rag — signal 3: self-consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestAbstainingRagSignal3:
    def _run(self, disagrees: bool, sim: float = 0.50) -> Result:
        with patch("src.router.retriever") as r, \
             patch("src.router.llm") as mock_llm, \
             patch("src.router.uncertainty") as u, \
             patch("src.router.ENABLE_SELF_CONSISTENCY", True):
            r.search.return_value = STRONG_HITS
            mock_llm.generate.return_value = GEN_OK
            u.retrieval_gate.return_value = (True, 0.70)
            u.support.return_value = SUP_OK
            u.disagrees_across_subsets.return_value = (disagrees, sim)
            return abstaining_rag("Q?")

    def test_abstains_when_self_consistency_fires(self) -> None:
        result = self._run(disagrees=True, sim=0.50)
        assert result.decision == "ABSTAIN"
        assert "Self-consistency" in result.reason

    def test_answers_when_self_consistency_passes(self) -> None:
        result = self._run(disagrees=False, sim=0.90)
        assert result.decision == "ANSWER"

    def test_consistency_sim_in_signals(self) -> None:
        result = self._run(disagrees=False, sim=0.88)
        assert result.signals["consistency_sim"] == pytest.approx(0.88)
