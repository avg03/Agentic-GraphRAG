"""Context Resolver (v2 §10.2, between planner and supervisor).

Deterministic, zero LLM: resolves the planner's candidate context keys and
scope root against the offline context index into ranked candidate roots,
and records them in state for the supervisor's GRAPH_SCOPE decisions.
"""

from __future__ import annotations

__all__ = ["resolve_candidate_contexts"]


def resolve_candidate_contexts(
    candidate_keys: list[str],
    scope_root: str = "",
    max_roots: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Candidate context keys -> (ranked roots, matched keys) for state.

    Roots: [{key, display, level, specificity, n_events, vertex_ids}...] —
    most specific first (v2 §3.4 specific-first expansion). Never raises:
    index problems degrade to "no candidate contexts".
    """
    phrases = [k for k in candidate_keys if k]
    if scope_root and scope_root not in phrases:
        phrases.append(scope_root)
    if not phrases:
        return [], []

    try:
        from graphrag.context_index import load_context_index, match_contexts
        from graphrag.scope import resolve_scope

        index = load_context_index()
        matched = match_contexts(phrases, index)
        roots = resolve_scope(matched["matches"], max_roots=max_roots)
    except Exception:
        return [], []

    return roots, matched["matches"]
