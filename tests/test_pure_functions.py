"""Pure-function regression tests (no TigerGraph, no LLM, no network).

Locks the behavior of the deterministic layers that the v2 architecture
builds on: name normalization, answer scoring, entity-match acceptance
thresholds, context derivation, and scope ranking. Run: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ---------------------------------------------------------------------------
# graphrag.resolve.normalize_name
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_case_and_whitespace(self):
        from graphrag.resolve import normalize_name

        assert normalize_name("  United   States ") == "united states"

    def test_punctuation_stripped(self):
        from graphrag.resolve import normalize_name

        assert normalize_name("Men's K-2 1000m") == "men s k 2 1000m"

    def test_leading_article_dropped(self):
        from graphrag.resolve import normalize_name

        assert normalize_name("the United States") == "united states"
        assert normalize_name("A Test Event") == "test event"

    def test_idempotent(self):
        from graphrag.resolve import normalize_name

        once = normalize_name("Canoeing at the 2012 Summer Olympics")
        assert normalize_name(once) == once


# ---------------------------------------------------------------------------
# graphrag.entity_index fuzzy-accept rule (the 180k-scale guard)
# ---------------------------------------------------------------------------

class TestFuzzyAccept:
    def test_near_identical_cosine_accepted(self):
        from graphrag.entity_index import fuzzy_accept

        assert fuzzy_accept(0.95, 0.0)

    def test_high_cosine_with_overlap_accepted(self):
        from graphrag.entity_index import fuzzy_accept

        assert fuzzy_accept(0.85, 0.75)

    def test_bare_cosine_rejected(self):
        from graphrag.entity_index import fuzzy_accept

        # clears 0.80 but token overlap too weak — the race-walk trap
        assert not fuzzy_accept(0.85, 0.3)

    def test_low_cosine_rejected(self):
        from graphrag.entity_index import fuzzy_accept

        assert not fuzzy_accept(0.70, 0.9)


# ---------------------------------------------------------------------------
# graphrag.eval_run answer matching (word-boundary rule)
# ---------------------------------------------------------------------------

class TestAnswerMatching:
    def _scored(self, predicted: str, gold: list[str]) -> bool:
        from graphrag.eval_run import score_question

        result = {
            "answer": type("A", (), {
                "answer": predicted,
                "cited_chunk_ids": [],
                "evidence_sufficient": True,
                "confidence": 1.0,
            })(),
            "chunks": [],
            "metrics": {},
        }
        gold_q = {"qid": "t", "question": "q", "answer": gold, "gold_doc_ids": []}
        return score_question(result, gold_q)["answer_match"]

    def test_exact(self):
        assert self._scored("Rafaela Silva", ["Rafaela Silva"])

    def test_word_boundary_no_false_substring(self):
        # gold "20" must NOT match "2004" (the fixed scoring bug)
        assert not self._scored("2004", ["20"])

    def test_word_in_sentence(self):
        assert self._scored("the winner was Chen Ding", ["Chen Ding"])

    def test_case_insensitive(self):
        assert self._scored("rafaela silva", ["Rafaela Silva"])


# ---------------------------------------------------------------------------
# shared.metrics contract shape
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_record_and_summary(self):
        from shared.metrics import Metrics

        m = Metrics()
        m.incr("chunks", 3)
        with m.timer("stage"):
            pass
        m.record_llm({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        s = m.summary()
        assert s["counters"]["chunks"] == 3
        assert s["llm"]["total_tokens"] == 15
        assert "stage" in s["timings"]


# ---------------------------------------------------------------------------
# shared.result metrics contract
# ---------------------------------------------------------------------------

class TestMetricsContract:
    def test_core_fields_validated(self):
        from shared.result import validate_metrics

        view = {
            "context_tokens": 1, "input_tokens": 1, "output_tokens": 1,
            "total_tokens": 1, "latency_ms": 1.0, "retrieval_latency_ms": 1.0,
            "llm_latency_ms": 1.0, "n_chunks": 1, "n_citations": 1,
            "evidence_sufficient": True, "confidence": 0.9,
        }
        assert validate_metrics(view, "vector_rag")["ok"]

    def test_agentic_extras_required(self):
        from shared.result import validate_metrics

        view = {k: 1 for k in (
            "context_tokens", "input_tokens", "output_tokens", "total_tokens",
            "latency_ms", "retrieval_latency_ms", "llm_latency_ms",
            "n_chunks", "n_citations", "evidence_sufficient", "confidence")}
        verdict = validate_metrics(view, "agentic_graphrag")
        assert not verdict["ok"]
        assert "steps" in verdict["missing_extras"]
        assert "contexts_investigated" in verdict["missing_extras"]

    def test_graphrag_extras_required(self):
        from shared.result import validate_metrics

        view = {k: 1 for k in (
            "context_tokens", "input_tokens", "output_tokens", "total_tokens",
            "latency_ms", "retrieval_latency_ms", "llm_latency_ms",
            "n_chunks", "n_citations", "evidence_sufficient", "confidence")}
        verdict = validate_metrics(view, "graph_rag")
        assert not verdict["ok"]
        assert "retrieval_mode" in verdict["missing_extras"]
        assert "scope_coverage" in verdict["missing_extras"]

    def test_vector_pipeline_has_no_extras(self):
        from shared.result import validate_metrics

        view = {k: 1 for k in (
            "context_tokens", "input_tokens", "output_tokens", "total_tokens",
            "latency_ms", "retrieval_latency_ms", "llm_latency_ms",
            "n_chunks", "n_citations", "evidence_sufficient", "confidence")}
        assert validate_metrics(view, "vector_rag")["ok"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
