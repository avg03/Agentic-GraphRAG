"""V2-layer pure-function tests: context derivation, scope ranking, supervision.

Complements test_pure_functions.py (v1 behavior locks) with the v2
architecture additions. No TigerGraph, no LLM, no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ---------------------------------------------------------------------------
# graphrag.context.derive_context (deterministic hierarchy, v2 §3.2)
# ---------------------------------------------------------------------------

VOCAB = {"Canoeing", "Speed skating", "Athletics", "Sailing", "Weightlifting"}


class TestDeriveContext:
    def test_canonical_title(self):
        from graphrag.context import derive_context

        c = derive_context(
            "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres", VOCAB
        )
        assert c["root_context"] == "2012 Summer Olympics"
        assert c["context_level"] == "event"
        assert "olympics:2012" in c["context_keys"]
        assert "sport:canoeing" in c["context_keys"]
        assert any(k.startswith("event:") for k in c["context_keys"])

    def test_title_without_event_is_sport_level(self):
        from graphrag.context import derive_context

        c = derive_context("Swimming at the 2008 Summer Olympics", VOCAB)
        assert c["context_level"] == "sport"
        assert c["context_keys"] == ["olympics:2008", "sport:swimming"]

    def test_event_first_name_finds_real_sport(self):
        from graphrag.context import derive_context

        c = derive_context(
            "Men's 1000 metres speed skating competition at the 2006 Winter Olympics",
            VOCAB,
        )
        assert "sport:speed_skating" in c["context_keys"]
        assert "men_s_1000_metres_speed_skating" not in c["context_keys"]

    def test_round_qualifiers_stripped(self):
        from graphrag.context import derive_context

        a = derive_context("Heats of Men's 1000 metres at the 1994 Winter Olympics", VOCAB)
        b = derive_context("Men's 1000 metres Final A at 1994 Winter Olympics", VOCAB)
        assert "event:men_s_1000_metres" in a["context_keys"]
        assert a["context_keys"] == b["context_keys"]

    def test_non_olympics_fallback(self):
        from graphrag.context import derive_context

        c = derive_context("2006 Lebanon War", VOCAB)
        assert c["context_level"] == "edition"
        assert c["context_keys"][0].startswith("topic:")
        assert "year:2006" in c["context_keys"]

    def test_empty_input(self):
        from graphrag.context import derive_context

        assert derive_context("") is None

    def test_merge_fills_sport_from_title(self):
        from graphrag.context import derive_context, merge_contexts

        event_ctx = derive_context("Heats of Men's 1000 metres at the 1994 Winter Olympics", VOCAB)
        title_ctx = derive_context(
            "Speed skating at the 1994 Winter Olympics – Men's 1000 metres", VOCAB
        )
        merged = merge_contexts(event_ctx, title_ctx)
        assert "sport:speed_skating" in merged["context_keys"]
        assert "olympics:1994" in merged["context_keys"]


# ---------------------------------------------------------------------------
# graphrag.scope (specificity ranking + expansion ladder, v2 §3.4/§6.3)
# ---------------------------------------------------------------------------

class TestScope:
    def test_specificity_order(self):
        from graphrag.scope import key_specificity

        assert key_specificity("event:x") > key_specificity("sport:x")
        assert key_specificity("sport:x") > key_specificity("olympics:2008")

    def test_resolve_scope_most_specific_first(self):
        from graphrag.scope import resolve_scope

        matches = [
            {"key": "olympics:2008", "display": "2008 olympics", "vertex_ids": ["e1", "e2", "e3"]},
            {"key": "sport:rowing", "display": "rowing", "vertex_ids": ["e1"]},
            {"key": "event:x", "display": "x", "vertex_ids": ["e1"]},
        ]
        roots = resolve_scope(matches, max_roots=3)
        assert [r["key"] for r in roots] == ["event:x", "sport:rowing", "olympics:2008"]

    def test_resolve_scope_dedupes_and_caps(self):
        from graphrag.scope import resolve_scope

        matches = [
            {"key": "olympics:2008", "display": "2008 olympics", "vertex_ids": ["e1", "e2"]},
            {"key": "olympics:2008", "display": "2008 olympics", "vertex_ids": ["e3"]},
            {"key": "year:2008", "display": "2008", "vertex_ids": ["e4"]},
        ]
        roots = resolve_scope(matches, max_roots=1)
        assert len(roots) == 1
        assert roots[0]["key"] == "olympics:2008"
        assert roots[0]["vertex_ids"] == ["e1", "e2", "e3"]

    def test_ladder(self):
        from graphrag.scope import expansion_ladder

        assert expansion_ladder(["event:x"]) == ["sport", "olympics"]
        assert expansion_ladder(["sport:x"]) == ["olympics"]
        assert expansion_ladder(["olympics:2008"]) == []

    def test_dedupe_event_ids(self):
        from graphrag.scope import dedupe_event_ids

        assert dedupe_event_ids(["e1", "e2", "e1", "", "e2"]) == ["e1", "e2"]


# ---------------------------------------------------------------------------
# agentic.supervisor stop-reason taxonomy (v2 §19)
# ---------------------------------------------------------------------------

class TestStopReasons:
    def test_known_reasons_pass_through(self):
        from agentic.supervisor import _normalize_stop_reason

        assert _normalize_stop_reason("sufficient_evidence") == "sufficient_evidence"
        assert _normalize_stop_reason("budget_exhausted") == "budget_exhausted"

    def test_unknown_reasons_map(self):
        from agentic.supervisor import _normalize_stop_reason

        assert _normalize_stop_reason("all_slots_supported") == "sufficient_evidence"
        assert _normalize_stop_reason("weird_reason") == "no_useful_path"


# ---------------------------------------------------------------------------
# graphrag.rank scope-aware scoring helpers (v2 §8/§9)
# ---------------------------------------------------------------------------

class TestRankHelpers:
    def test_context_match(self):
        from graphrag.rank import _context_match

        assert _context_match(["olympics:2008"], ["sport:rowing", "olympics:2008"]) == 1.0
        assert _context_match(["sport:canoeing"], ["olympics:2008"]) == 0.0
        assert _context_match([], ["olympics:2008"]) == 0.0

    def test_coverage_collection(self):
        from graphrag.rank import _coverage

        assert _coverage(0, 0, "collection_aggregation") == "none"
        assert _coverage(2, 6, "collection_aggregation") == "partial"
        assert _coverage(6, 10, "collection_aggregation") == "substantial"
        assert _coverage(12, 20, "collection_aggregation") == "high"

    def test_coverage_local(self):
        from graphrag.rank import _coverage

        assert _coverage(1, 2, "local_fact") == "partial"
        assert _coverage(1, 5, "local_fact") == "high"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
