"""GraphRAG v2 orchestration (architecture v2 §4.1, §16-§17).

The single entry point for graph retrieval, exposed to both Pipeline 2 and
the agentic tools:

    graph_search(query, seeds=None, context_keys=None, mode="auto", top_k=...)

Separates DISCOVERY from TRAVERSAL (v2 §3.3):

    1. Query entity + scope analysis   (1 LLM call, unless analysis given)
    2. Entity matching                 (seeds, via entity_index)
    3. Context resolution              (candidate keys -> ranked roots,
                                        via context_index + scope)
    4. Mode dispatch                   (local | scope | auto)
    5. Traversal                       (Mode A seeds / Mode B sibling events,
                                        merged when both apply)
    6. Scope-aware ranking + gate      (rank.rank_chunks)

`auto` selects scope traversal when the analysis classifies the question as
collection_aggregation AND a context root resolved; otherwise local.
"""

from shared.metrics import Metrics
from shared.models import QueryAnalysis
from shared.prompts import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    build_query_analysis_prompt,
)

from graphrag import config
from graphrag.rank import COLLECTION_TOP_N, DEFAULT_TOP_N, rank_chunks
from graphrag.scope import dedupe_event_ids, resolve_scope

__all__ = ["graph_search", "analyze_query"]


def analyze_query(query: str, metrics: Metrics | None = None) -> QueryAnalysis:
    """One Gemini structured call: entities + question type + scope + keys."""
    from shared.llm_client import generate_structured

    parsed, usage = generate_structured(
        prompt=build_query_analysis_prompt(query),
        response_schema=QueryAnalysis,
        system=QUERY_ANALYSIS_SYSTEM_PROMPT,
    )
    if metrics is not None:
        latency = usage.pop("_latency_ms", None)
        metrics.record_llm(usage, latency_ms=latency)
        metrics.incr("query_entities", len(parsed.entities))
    return parsed


def _resolve_roots(context_keys: list[str], scope_root: str, metrics=None):
    """Candidate context keys/phrases -> ranked scope roots (deterministic)."""
    from graphrag.context_index import load_context_index, match_contexts

    phrases = [k for k in context_keys if k]
    if scope_root and scope_root not in phrases:
        phrases.append(scope_root)
    if not phrases:
        return [], []

    index = load_context_index()
    matched = match_contexts(phrases, index)
    roots = resolve_scope(matched["matches"])
    if metrics is not None:
        metrics.incr("context_roots", len(roots))
        metrics.incr("unmatched_context_phrases", len(matched["unmatched"]))
    return roots, matched["matches"]


def _candidate_events(roots: list[dict]) -> tuple[list[str], dict[str, list[str]]]:
    """Ranked roots -> capped, deduped candidate ActionEvent ids + lineage."""
    event_ids: list[str] = []
    event_to_keys: dict[str, list[str]] = {}
    per_root = config.SCOPE_MAX_EVENTS_PER_ROOT
    for root in roots:
        for eid in root["vertex_ids"][:per_root]:
            event_ids.append(eid)
            event_to_keys.setdefault(eid, []).append(root["key"])
    return dedupe_event_ids(event_ids), event_to_keys


def _merge_chunks(base: dict, extra: dict) -> dict:
    """Merge two {chunk_id: entry} traversal maps (support max-summed,
    via/event_ids unioned)."""
    for cid, entry in extra.items():
        if cid not in base:
            base[cid] = entry
            continue
        tgt = base[cid]
        tgt["support"] += entry.get("support", 0)
        for via in entry.get("via", []):
            if via not in tgt["via"]:
                tgt["via"].append(via)
        for eid in entry.get("event_ids", []):
            if eid not in tgt["event_ids"]:
                tgt["event_ids"].append(eid)
    return base


def graph_search(
    query: str,
    seeds: list[tuple[str, str, str]] | None = None,
    context_keys: list[str] | None = None,
    mode: str = "auto",
    top_k: int = DEFAULT_TOP_N,
    collection=None,
    conn=None,
    analysis: QueryAnalysis | None = None,
    metrics: Metrics | None = None,
    allow_dense_topup: bool = True,
    max_paths: int = 20,
) -> dict:
    """V2 graph retrieval: entity seeds + context scope -> ranked chunks.

    Args:
        query: The user question (or an agent's subquery).
        seeds: Pre-matched seed vertices [(vtype, vid, name)]; matched from
            the query when omitted.
        context_keys: Explicit scope keys ("olympics:2008"); when omitted,
            candidates come from the query analysis.
        mode: "local" (Mode A) | "scope" (Mode B) | "auto".
        top_k: Final chunk budget (raised for collection questions).
        analysis: Reuse an existing QueryAnalysis (skips the LLM call).
    """
    metrics = metrics or Metrics()
    context_keys = list(context_keys or [])

    # 1. query analysis (entities + question type + scope)
    with metrics.timer("stage1_query_analysis"):
        if analysis is None:
            analysis = analyze_query(query, metrics)
    question_type = analysis.question_type or "local_fact"

    # 2. entity matching -> seeds
    with metrics.timer("stage2_entity_matching"):
        if seeds is None:
            from graphrag.entity_index import load_entity_index, match_entities

            if conn is None:
                from graphrag.schema import connect

                conn = connect()
            index = load_entity_index(conn)
            mentions = [e.model_dump() for e in analysis.entities]
            # the raw query is also a match candidate: canonical event names
            # live in vertex names (see agentic.tools.graph_search v1)
            mentions.append({"name": query[:220], "type": None})
            matching = match_entities(mentions, index)
            seeds = matching["seeds"]
        else:
            matching = {"seeds": seeds, "resolved": [], "unmatched": []}
    metrics.incr("seed_entities", len(seeds))

    # 3. context resolution -> ranked roots
    with metrics.timer("stage3_context_resolution"):
        roots, matched_keys = _resolve_roots(
            context_keys + analysis.candidate_context_keys,
            analysis.scope.root if analysis.scope else "",
            metrics,
        )
    requested_keys = [r["key"] for r in roots] or [m["key"] for m in matched_keys]

    # 4. mode selection (v2 §17: auto picks from the question type)
    mode_used = mode
    if mode == "auto":
        mode_used = (
            "scope"
            if question_type == "collection_aggregation" and roots
            else "local"
        )
    if mode_used == "scope" and not roots:
        mode_used = "local"  # no resolvable scope: local traversal only
    metrics.counters["retrieval_mode"] = mode_used

    if conn is None:
        from graphrag.schema import connect

        conn = connect()

    # 5. traversal (Mode A / Mode B; merged when seeds exist in scope mode)
    with metrics.timer("stage4_traverse"):
        if mode_used == "scope":
            candidate_events, event_to_keys = _candidate_events(roots)
            metrics.incr("scope_candidate_events", len(candidate_events))
            chunk_entries = traverse_scope_events(conn, candidate_events)
            for entry in chunk_entries.values():
                keys: list[str] = []
                for eid in entry.get("event_ids", []):
                    for key in event_to_keys.get(eid, []):
                        if key not in keys:
                            keys.append(key)
                entry["context_keys"] = keys
            if seeds:
                from graphrag.traverse import traverse

                chunk_entries = _merge_chunks(chunk_entries, traverse(conn, seeds))
        else:
            from graphrag.traverse import traverse

            chunk_entries = traverse(conn, seeds) if seeds else {}
            for entry in chunk_entries.values():
                entry.setdefault("context_keys", [])
    metrics.incr("graph_chunks", len(chunk_entries))

    # relationship paths for the context builder / agent evidence
    with metrics.timer("stage4b_relationships"):
        relationships = _collect_relationships(conn, mode_used, seeds, roots, max_paths)

    # 6. scope-aware ranking + query-aware gate
    if question_type == "collection_aggregation":
        top_k = max(top_k, COLLECTION_TOP_N)
    with metrics.timer("stage5_rank"):
        ranked, rank_info = rank_chunks(
            query,
            chunk_entries,
            top_n=top_k,
            collection=collection,
            allow_dense_topup=allow_dense_topup,
            question_type=question_type if roots or mode_used == "scope" else None,
            requested_keys=requested_keys,
            retrieval_mode=mode_used,
        )
    metrics.incr("dense_topup_chunks", rank_info["dense_topup"])
    metrics.incr("final_chunks", len(ranked))

    return {
        "chunks": ranked,
        "seeds": seeds,
        "matching": matching,
        "rank_info": rank_info,
        "analysis": analysis.model_dump(),
        "question_type": question_type,
        "retrieval_mode": mode_used,
        "scope_roots": roots,
        "scope_matches": matched_keys,
        "relationships": relationships,
        "metrics": metrics,
    }


def traverse_scope_events(conn, event_ids: list[str]) -> dict:
    """Mode B traversal wrapper (kept separate for agent tool reuse)."""
    from graphrag.traverse import traverse_scope

    return traverse_scope(conn, event_ids)


def _collect_relationships(conn, mode_used: str, seeds, roots, max_paths: int) -> list[str]:
    """Human-readable relationship paths for evidence display."""
    from graphrag.traverse import collect_relationships

    if mode_used == "scope":
        top_events = dedupe_event_ids(
            [eid for root in roots for eid in root["vertex_ids"]]
        )[:10]
        event_seeds = [("ActionEvent", eid, eid) for eid in top_events]
        return collect_relationships(conn, event_seeds, max_paths=max_paths)
    if seeds:
        return collect_relationships(conn, seeds, max_paths=max_paths)
    return []
