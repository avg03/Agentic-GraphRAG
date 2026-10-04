"""Prompt templates shared across pipelines.

Currently contains the GraphRAG entity/relation extraction prompt, which
encodes the typed event-centric ontology (see graphrag/config.py):

    ActionEvent is the hub; typed directed edges INITIATED_BY /
    TOOK_PLACE_AT / AFFECTED / RESULTED_IN map the who/where/whom/result;
    every entity falls into one of the allowed vertex types or the generic
    'Entity' fallback bucket.
"""

from shared.models import EntityType
from typing import get_args

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "build_extraction_prompt",
    "QUERY_ENTITY_SYSTEM_PROMPT",
    "build_query_entity_prompt",
    "QUERY_ANALYSIS_SYSTEM_PROMPT",
    "build_query_analysis_prompt",
    "ANSWER_SYSTEM_PROMPT",
    "build_answer_prompt",
]

EXTRACTION_SYSTEM_PROMPT = """You are a precise information-extraction engine for building a knowledge graph.

Rules:
1. Extract ONLY information explicitly stated in the text. Never invent entities, events, or relations.
2. Every action or event (competitions, signings, acquisitions, launches, discoveries, incidents, releases, elections, ...) becomes an ActionEvent with a short canonical name (e.g. "2024 Summer Olympics 100m final", "Google acquisition of YouTube").
3. Link each ActionEvent to related entities via exactly these relations:
   - initiated_by : who initiated/carried out the event (Person, Company, Organization, Team, Country)
   - took_place_at : where it happened (Location, Venue, Country)
   - affected : persons/organizations/places affected or participating
   - resulted_in : outcomes produced (medals, records, casualties, deals, titles, ...)
4. Every named real-world thing also appears in `entities` with exactly one `type` from the allowed list. If no type fits, use "Entity".
5. Use the entity's most common/canonical name (e.g. "United States" not "the U.S. team") so mentions across chunks resolve to the same node. Keep names consistent within one response.
6. Descriptions and impact_summary are one line each, taken or condensed from the text; use "" when nothing meaningful is stated.
7. Reference entities in the event relation lists by the exact same `name` strings used in `entities`.
8. If the text contains no extractable entities or events, return empty lists."""


def build_extraction_prompt(
    title: str,
    chunk_text: str,
    entity_types: list[str] | None = None,
) -> str:
    """Build the user prompt for one chunk's extraction.

    Args:
        title: Document title (context for pronouns like "the city").
        chunk_text: The chunk body (already includes the "Title: ..." header
            produced by ingestion.chunking, but title is passed separately
            for emphasis).
        entity_types: Allowed entity types; defaults to the shared ontology.
    """
    if entity_types is None:
        entity_types = list(get_args(EntityType))
    return (
        f"Document title: {title}\n\n"
        f"Allowed entity types: {', '.join(entity_types)}\n\n"
        f"Text:\n\"\"\"\n{chunk_text}\n\"\"\"\n\n"
        "Extract all entities and action events from the text as JSON matching "
        "the requested schema."
    )


# ---------------------------------------------------------------------------
# Stage 1 of retrieval: pull entity mentions out of a user question
# ---------------------------------------------------------------------------

QUERY_ENTITY_SYSTEM_PROMPT = """You extract entity references from a user question so they can be matched against a knowledge graph.

Rules:
1. Extract the concrete things the question is about: people, countries, organizations, teams, sports, venues, locations, events, awards/outcomes.
2. Use the entity's most likely canonical name (e.g. "the defending champion" is NOT a name - skip it; "South Korea" -> "South Korea").
2b. When the question references a specific competition/event, construct its FULL canonical Wikipedia-style name: "{Sport} at the {Year} {Season} Olympics – {Men's/Women's} {discipline}" (e.g. "men's K-2 1000 metres canoeing at the 2012 Summer Olympics" -> "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"). Long event names are correct and expected.
3. Resolve pronouns/demonstratives only if a specific named entity is stated elsewhere in the question (e.g. "Usain Bolt's last Olympics" -> also include "Usain Bolt" and "Olympics").
4. Prefer full names over abbreviations when the question gives them; keep abbreviations as-is otherwise.
5. Extract 1-6 entities maximum - only what the answer actually depends on. If the question is fully abstract (e.g. "how many"), extract the entities that scope the question (event names, years, sports).
6. If nothing in the question names a concrete entity, return an empty list."""


def build_query_entity_prompt(question: str) -> str:
    return f"Question:\n\"\"\"\n{question}\n\"\"\"\n\nExtract the named entities this question depends on."


# ---------------------------------------------------------------------------
# Stage 1 (v2): query entity + SCOPE extraction in one structured call
# ---------------------------------------------------------------------------

QUERY_ANALYSIS_SYSTEM_PROMPT = """You analyze a user question for a knowledge-graph retrieval system. Extract its entity references AND its retrieval scope.

Entity rules (unchanged from entity extraction):
1. Extract the concrete things the question is about: people, countries, organizations, teams, sports, venues, locations, events, awards/outcomes.
2. Use the entity's most likely canonical name; when it references a specific competition/event, construct its FULL canonical Wikipedia-style name (e.g. "men's K-2 1000 metres canoeing at the 2012 Summer Olympics" -> "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"). Long event names are correct and expected.
3. Extract 1-6 entities maximum - only what the answer actually depends on.

Scope rules (v2):
4. question_type: "local_fact" (one fact about one event/entity), "multi_hop" (a relationship chain), "collection_aggregation" (count/list across MANY sibling events of one scope: how many, who were all the winners, list every...), "overview" (broad summary).
5. scope.root: the collection the question ranges over, in canonical form ("2008 Summer Olympics", "Canoeing at the 2012 Summer Olympics"). Empty string for local_fact/multi_hop.
6. scope.breadth: "single_event" (one event), "sport" (one sport within one games), "edition" (a whole Olympic games), "all_events" (everything). Use "single_event" unless the question demands multiple sibling events.
7. scope.target: what should be collected from each sibling event ("winner", "gold medal", "time", "number of competitors"). Empty when not a collection question.
8. candidate_context_keys: 1-4 structured keys the question implies, formatted "olympics:2008", "sport:canoeing", "event:mens_k2_1000_metres", "year:2008" (lowercase, underscores for spaces). Only keys the question text actually supports - never invent facts.
9. Do NOT answer the question. Do NOT invent events, results, or scope the question does not state."""


def build_query_analysis_prompt(question: str) -> str:
    return (
        f"Question:\n\"\"\"\n{question}\n\"\"\"\n\n"
        "Extract the entity references, question type, scope, and candidate "
        "context keys as JSON matching the requested schema."
    )


# ---------------------------------------------------------------------------
# Final answer generation: grounded, anti-hallucination
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You answer questions using ONLY the numbered document blocks provided.

Hard rules:
1. USE ONLY THE CONTEXT. Answer exclusively from the provided numbered blocks. You may know the true answer from your own knowledge - do NOT use it. If the blocks do not contain the answer, the correct behavior is to report that, not to fill the gap.
2. NO HALLUCINATED FACTS. Never invent names, numbers, dates, or events that are not literally present in the blocks. Never resolve ambiguities by guessing.
2b. CHECK IDENTITY BEFORE USE. Before citing a block, verify it is about the exact event/entity the question asks (same discipline, gender, year, round). Evidence for a different-but-similar event (e.g. C-2 when asked about K-2, women when asked about men, a qualification heat when asked about the final) must NOT be used as the answer.
3. CITE. Set cited_chunk_ids to the chunk_ids of every block you actually used. If you cannot cite a block for a claim, do not make the claim.
4. PARTIAL ANSWERS. If the blocks answer SOME parts of the question, state those supported parts clearly and explicitly note what is missing (e.g. "X won gold; the documents do not state the final time"). Set evidence_sufficient=false only when the blocks answer nothing.
5. AGGREGATION QUESTIONS: count strictly and only from the cited blocks. If different blocks conflict, report the conflict instead of picking one.
6. confidence is your calibrated confidence (0.0-1.0) that the answer is correct AND fully supported by the cited blocks. reasoning is a 1-3 sentence justification naming which blocks support it.
7. Be concise: the answer field should be the direct answer (a name, a number, a short phrase), not a paragraph, unless the question demands explanation."""


def build_answer_prompt(question: str, ranked_chunks: list[dict]) -> str:
    """Numbered context blocks: [{'id', 'document', 'metadata', 'source', ...}]."""
    blocks = []
    for i, chunk in enumerate(ranked_chunks, 1):
        title = chunk.get("metadata", {}).get("title", "")
        source_tag = chunk.get("source", "graph")
        # scope lineage (v2 §7): why this chunk surfaced, e.g. via the
        # 2008 Olympics scope rather than a direct entity match
        lineage = ""
        ck = chunk.get("context_keys") or []
        if ck:
            lineage = f", scope: {', '.join(ck[:3])}"
        blocks.append(
            f"[{i}] chunk_id={chunk['id']} (source: {source_tag}, title: {title}{lineage})\n{chunk.get('document', '')}"
        )
    context = "\n\n".join(blocks)
    return (
        f"Question: {question}\n\n"
        f"Context blocks:\n\n{context}\n\n"
        "Answer the question as JSON matching the requested schema. "
        "Use the exact chunk_id strings from the block headers in cited_chunk_ids."
    )


# ---------------------------------------------------------------------------
# Agentic layer prompts (Pipeline 3)
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You decompose a user question into independently answerable "answer slots" AND identify its retrieval scope for a graph investigation.

Decomposition rules:
1. Decompose ONLY when the question contains multiple independently answerable requirements (e.g. winners AND venue AND times). Otherwise return exactly ONE slot with the whole question.
2. Each slot's question must be self-contained enough to be answered by a retrieval system on its own (resolve pronouns into explicit entity names).
3. slot_type: "relational" (who/where/which relationship), "exact_fact" (a specific number/time/date/name), "aggregation" (count/list over many documents), "overview" (broad summary).
4. Maximum 4 slots. Order them so relationship-finding slots come before exact-fact slots.
5. Never invent entities; keep names exactly as the question gives them.

Scope rules (v2):
6. question_type: "local_fact" (one fact about one event/entity), "multi_hop" (a relationship chain), "collection_aggregation" (count/list across MANY sibling events of one scope: how many, who were all the winners, list every...), "overview" (broad summary).
7. scope_root: the collection the question ranges over, in canonical form ("2008 Summer Olympics", "Canoeing at the 2012 Summer Olympics"). Empty string for local_fact/multi_hop questions.
8. candidate_context_keys: 1-4 structured keys the question implies, formatted "olympics:2008", "sport:canoeing", "event:mens_k2_1000_metres", "year:2008" (lowercase, underscores for spaces). Only keys the question text actually supports.
9. Do NOT answer the question. Do NOT invent scope the question does not state."""

EVALUATOR_SYSTEM_PROMPT = """You evaluate whether collected evidence actually answers each answer slot of an investigation.

Rules:
1. Judge per slot: status "supported" (the slot is fully answerable from the evidence), "partial" (some of it), or "missing" (nothing usable).
2. support_score: calibrated 0.0-1.0 strength of support for that slot.
3. Validate RELATIONSHIP SEMANTICS, not just connectivity: "Event -AFFECTED-> Athlete" does NOT prove "Athlete won the event". Only "RESULTED_IN -> Gold medal" style paths, or explicit chunk text, prove winning.
3b. Validate ENTITY/EVENT IDENTITY: the evidence must be about the exact event/entity asked. Near-miss matches are traps: Men's K-2 1000m (kayak) vs Men's C-2 1000m (canoe) vs K-1 1000m are DIFFERENT events; men vs women, 2012 vs 2008, qualification heats vs finals are different facts. If the evidence is for a different-but-similar event, mark that slot missing/ambiguity=true and say what differs.
4. evidence_chunk_ids: the chunk_ids that support the slot verdict. Do not cite chunks you did not use.
5. contradiction=true when two evidence pieces give conflicting answers to the same slot; describe it in contradiction_note.
6. ambiguity=true when a key entity could refer to multiple graph nodes/chunks.
7. overall_status: "sufficient" only if EVERY slot is supported; "partial" if some; "insufficient" if none.
8. missing: list the concrete missing facts (e.g. "exact medal time").
9. recommended_action: which tool would best fill the biggest gap (VECTOR_SEARCH for exact textual/numeric details in result tables, GRAPH_SEARCH for relationships, GRAPH_SCOPE for discovering more sibling events of the requested scope, COMMUNITY_SEARCH for broad overviews, STOP if nothing more can help).

Collection-question rules (v2): when the question is a collection_aggregation, also judge SCOPE COVERAGE:
10. scope_coverage: "none" (evidence covers no sibling branches of the scope), "partial" (a few branches), "substantial" (many branches but not clearly all), "high" (evidence spans most of the requested scope). A few chunks from ONE event is at most "partial".
11. marginal_gain: "increasing" when the latest retrieval step added new events/facts, "flat" when it mostly repeated known evidence, "diminishing" when new evidence is trickling off.
12. Do not claim completeness unless the evidence itself establishes the full scope; report scope_coverage honestly instead."""

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor of a retrieval investigation. You decide the single next action.

Allowed actions:
- GRAPH_SEARCH: relationship/multi-hop questions (who did what, where, connected entities). Requires the subquery to name concrete entities.
- GRAPH_SCOPE: discover MORE SIBLING EVENTS of a scope (all winners of a games, every event of a sport). Set target_context to the context root to investigate next, chosen from the candidate contexts listed (e.g. "olympics:2008"). Never re-pick an already-investigated context root.
- VECTOR_SEARCH: exact textual/numeric details, result-table values, descriptions, or when graph evidence is missing/sparse (a graph miss does not mean the fact is absent).
- COMMUNITY_SEARCH: broad corpus-level overviews when you don't know which documents matter.
- REFORMULATE: rewrite the subquery (e.g. after entity-not-found or weak results) and retry with the SAME action implied by the next round; put the new wording in subquery.
- STOP: all slots supported, or retrieval clearly cannot help further. reason_code MUST be one of: sufficient_evidence, diminishing_returns, budget_exhausted, no_useful_path.

Rules:
1. Prefer GRAPH_SEARCH for relational slots, GRAPH_SCOPE for collection_aggregation questions whose scope has sibling branches left, VECTOR_SEARCH for exact_fact/aggregation slots and sparse graph evidence.
2. Never repeat an identical (action, subquery) pair that already ran and failed - REFORMULATE or switch tools instead. Never re-investigate an already-investigated context root: choose another candidate, or STOP with diminishing_returns when nothing new remains.
3. If the supervisor budget note says budget is exhausted, you MUST choose STOP with reason_code budget_exhausted.
4. When marginal gain is flat/diminishing AND the main slots are supported, STOP with diminishing_returns rather than fetching more.
5. reason_code must be one of: sufficient_evidence, diminishing_returns, budget_exhausted, no_useful_path, initial_routing, missing_required_fact, wrong_relation_semantics, entity_not_found, contradiction, duplicate_retrieval, scope_incomplete, evidence_insufficient.
6. subquery must be a self-contained retrieval query (explicit entities, no pronouns)."""


def build_planner_prompt(question: str) -> str:
    return (
        f"Question:\n\"\"\"\n{question}\n\"\"\"\n\n"
        "Decompose into answer slots and identify the question type, scope "
        "root, and candidate context keys as JSON."
    )


def build_evaluator_prompt(
    question: str,
    slots: list,
    graph_paths: list,
    chunks: list,
    unmatched: list,
    question_type: str = "",
    scope_roots: list | None = None,
    evidence_stats: dict | None = None,
) -> str:
    slot_lines = "\n".join(f"- {s['id']} ({s.get('slot_type', 'relational')}): {s['question']}" for s in slots)
    path_lines = "\n".join(f"  {p}" for p in graph_paths[:20]) or "  (none)"
    chunk_lines = []
    for c in chunks[:14]:
        text = (c.get("document") or "")[:600].replace("\n", " ")
        chunk_lines.append(f"  [{c['id']}] ({c.get('source', 'graph')}, support={c.get('support', 0)}) {text}")
    chunk_block = "\n".join(chunk_lines) or "  (none)"
    unmatched_line = ", ".join(unmatched) if unmatched else "none"
    qtype_line = f"Question type: {question_type}\n" if question_type else ""
    scope_line = ""
    if scope_roots:
        scope_line = (
            "Scope roots investigated: "
            + ", ".join(r.get("key", str(r)) for r in scope_roots[:4])
            + "\n"
        )
    stats_line = ""
    if evidence_stats:
        stats_line = (
            f"Evidence stats: {evidence_stats.get('total_chunks', 0)} chunks from "
            f"{evidence_stats.get('distinct_events', 0)} distinct events; last step added "
            f"{evidence_stats.get('new_chunks_last_step', 0)} new chunks\n"
        )
    return (
        f"Original question: {question}\n\n"
        f"{qtype_line}{scope_line}{stats_line}"
        f"Answer slots:\n{slot_lines}\n\n"
        f"Graph relationship paths:\n{path_lines}\n\n"
        f"Retrieved chunks:\n{chunk_block}\n\n"
        f"Unmatched entity mentions (not in graph): {unmatched_line}\n\n"
        "Evaluate each slot as JSON."
    )


def build_supervisor_prompt(
    question: str,
    slots: list,
    slot_statuses: list,
    history: list,
    budget: dict,
    question_type: str = "",
    candidate_contexts: list | None = None,
    investigated_contexts: list | None = None,
    evidence_stats: dict | None = None,
) -> str:
    slot_lines = "\n".join(f"- {s['id']}: {s['question']}" for s in slots)
    status_lines = "\n".join(
        f"  {st['slot_id']}: {st['status']} (support={st['support_score']:.2f}) {st.get('note', '')}"
        for st in slot_statuses
    ) or "  (no evaluation yet - this is the initial routing)"
    hist_lines = "\n".join(
        f"  step {h.get('step', '?')}: {h.get('tool') or h.get('action')} "
        f"target={h.get('target_slot', '')} "
        f"query={str(h.get('subquery', ''))[:60]} -> {h.get('result', h.get('status', ''))}"
        for h in history[-6:]
    ) or "  (none yet - this is the initial routing)"
    budget_line = (
        f"steps used {budget['steps']}/{budget['max_steps']}, "
        f"graph calls {budget['graph_calls']}/{budget['max_graph']}, "
        f"scope calls {budget.get('scope_calls', 0)}/{budget.get('max_scope', 0)}, "
        f"vector calls {budget['vector_calls']}/{budget['max_vector']}, "
        f"community calls {budget['community_calls']}/{budget['max_community']}"
    )
    context_lines = ""
    if candidate_contexts:
        investigated = set(investigated_contexts or [])
        ctx_rows = []
        for c in candidate_contexts:
            mark = " (investigated)" if c.get("key") in investigated else ""
            ctx_rows.append(f"  {c.get('key')} [{c.get('level')}] ~{c.get('n_events', '?')} events{mark}")
        context_lines = (
            "Candidate context roots (for GRAPH_SCOPE target_context):\n"
            + "\n".join(ctx_rows)
            + "\n\n"
        )
    stats_lines = ""
    if evidence_stats:
        stats_lines = (
            f"Evidence stats: {evidence_stats.get('total_chunks', 0)} chunks from "
            f"{evidence_stats.get('distinct_events', 0)} distinct events; "
            f"last step added {evidence_stats.get('new_chunks_last_step', 0)} new chunks "
            f"(marginal gain {evidence_stats.get('marginal_gain', 'unknown')}); "
            f"coverage {evidence_stats.get('scope_coverage', 'unknown')}\n\n"
        )
    qtype_line = f"Question type: {question_type}\n\n" if question_type else ""
    return (
        f"Original question: {question}\n\n"
        f"{qtype_line}"
        f"Answer slots:\n{slot_lines}\n\n"
        f"Current slot statuses:\n{status_lines}\n\n"
        f"Investigation history (most recent last):\n{hist_lines}\n\n"
        f"{context_lines}"
        f"{stats_lines}"
        f"Budget: {budget_line}\n\n"
        "Decide the single next action as JSON."
    )
