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
        blocks.append(f"[{i}] chunk_id={chunk['id']} (source: {source_tag}, title: {title})\n{chunk.get('document', '')}")
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

PLANNER_SYSTEM_PROMPT = """You decompose a user question into independently answerable "answer slots" for a retrieval investigation.

Rules:
1. Decompose ONLY when the question contains multiple independently answerable requirements (e.g. winners AND venue AND times). Otherwise return exactly ONE slot with the whole question.
2. Each slot's question must be self-contained enough to be answered by a retrieval system on its own (resolve pronouns into explicit entity names).
3. slot_type: "relational" (who/where/which relationship), "exact_fact" (a specific number/time/date/name), "aggregation" (count/list over many documents), "overview" (broad summary).
4. Maximum 4 slots. Order them so relationship-finding slots come before exact-fact slots.
5. Never invent entities; keep names exactly as the question gives them."""

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
9. recommended_action: which tool would best fill the biggest gap (VECTOR_SEARCH for exact textual/numeric details in result tables, GRAPH_SEARCH for relationships, COMMUNITY_SEARCH for broad overviews, STOP if nothing more can help)."""

SUPERVISOR_SYSTEM_PROMPT = """You are the supervisor of a retrieval investigation. You decide the single next action.

Allowed actions:
- GRAPH_SEARCH: relationship/multi-hop questions (who did what, where, connected entities). Requires the subquery to name concrete entities.
- VECTOR_SEARCH: exact textual/numeric details, result-table values, descriptions.
- COMMUNITY_SEARCH: broad corpus-level overviews when you don't know which documents matter.
- REFORMULATE: rewrite the subquery (e.g. after entity-not-found or weak results) and retry with the SAME action implied by the next round; put the new wording in subquery.
- STOP: all slots supported, or the budget is exhausted, or more retrieval clearly cannot help.

Rules:
1. Prefer GRAPH_SEARCH for relational slots and VECTOR_SEARCH for exact_fact/aggregation slots. Do not call a tool for a slot that is already supported.
2. Never repeat an identical (action, subquery) pair that already ran and failed - REFORMULATE or switch tools instead.
3. If the supervisor budget note says budget is exhausted, you MUST choose STOP with reason_code budget_exhausted.
4. reason_code must be one of: initial_routing, missing_required_fact, wrong_relation_semantics, entity_not_found, contradiction, duplicate_retrieval, budget_exhausted, all_slots_supported, evidence_insufficient.
5. subquery must be a self-contained retrieval query (explicit entities, no pronouns)."""


def build_planner_prompt(question: str) -> str:
    return f"Question:\n\"\"\"\n{question}\n\"\"\"\n\nDecompose into answer slots as JSON."


def build_evaluator_prompt(
    question: str,
    slots: list,
    graph_paths: list,
    chunks: list,
    unmatched: list,
) -> str:
    slot_lines = "\n".join(f"- {s['id']} ({s.get('slot_type', 'relational')}): {s['question']}" for s in slots)
    path_lines = "\n".join(f"  {p}" for p in graph_paths[:20]) or "  (none)"
    chunk_lines = []
    for c in chunks[:14]:
        text = (c.get("document") or "")[:600].replace("\n", " ")
        chunk_lines.append(f"  [{c['id']}] ({c.get('source', 'graph')}, support={c.get('support', 0)}) {text}")
    chunk_block = "\n".join(chunk_lines) or "  (none)"
    unmatched_line = ", ".join(unmatched) if unmatched else "none"
    return (
        f"Original question: {question}\n\n"
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
        f"vector calls {budget['vector_calls']}/{budget['max_vector']}, "
        f"community calls {budget['community_calls']}/{budget['max_community']}"
    )
    return (
        f"Original question: {question}\n\n"
        f"Answer slots:\n{slot_lines}\n\n"
        f"Current slot statuses:\n{status_lines}\n\n"
        f"Investigation history (most recent last):\n{hist_lines}\n\n"
        f"Budget: {budget_line}\n\n"
        "Decide the single next action as JSON."
    )
