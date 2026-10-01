"""Shared Pydantic models for extraction outputs.

These serve both purposes:
  1. `response_schema` for Gemini structured output (google-genai accepts
     Pydantic models directly), and
  2. validated parse targets when reading cached extraction JSONL.

Every extracted item is grounded to a chunk via `chunk_id`, which is set by
the extraction step (Gemini never sees/returns it — the driver attaches it).
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

__all__ = [
    "ExtractedEntity",
    "ExtractedActionEvent",
    "ChunkExtraction",
    "QueryEntities",
    "GroundedAnswer",
    "AnswerSlot",
    "PlanOutput",
    "SlotStatus",
    "EvidenceEvaluation",
    "SupervisorDecision",
]

# Must match graphrag.config.ENTITY_TYPES (kept literal here so Gemini's
# response schema carries a hard enum instead of a free-form string).
EntityType = Literal[
    "Person",
    "Company",
    "Organization",
    "Team",
    "Country",
    "Location",
    "Venue",
    "Sport",
    "Outcome",
    "Entity",
]


class ExtractedEntity(BaseModel):
    """A non-event vertex mentioned in the chunk."""

    name: str = Field(description="Canonical name of the entity as written in the text.")
    type: EntityType = Field(
        description=(
            "One of the allowed vertex types. Use 'Entity' only when no other "
            "type fits."
        )
    )
    description: str = Field(default="", description="One-line description from the text (may be empty).")


class ExtractedActionEvent(BaseModel):
    """The central hub for an action: who initiated it, where, whom it affected, result."""

    name: str = Field(description="Short canonical name for the event/action, e.g. '2024 Summer Olympics 100m final'.")
    action_type: str = Field(description="Free-form category, e.g. 'competition', 'acquisition', 'discovery'.")
    description: str = Field(default="", description="One-line description of the event.")
    impact_summary: str = Field(default="", description="Short impact/consequence summary (may be empty).")
    initiated_by: List[str] = Field(
        default_factory=list,
        description="Names of Person/Company/Organization/Team/Country that initiated or carried out the event.",
    )
    took_place_at: List[str] = Field(
        default_factory=list,
        description="Names of Location/Venue/Country where the event took place.",
    )
    affected: List[str] = Field(
        default_factory=list,
        description="Names of Person/Company/Organization/Team/Country/Location affected by the event.",
    )
    resulted_in: List[str] = Field(
        default_factory=list,
        description="Names of Outcome vertices produced by the event (medals, records, casualties, deals, ...).",
    )


class ChunkExtraction(BaseModel):
    """Structured extraction result for one chunk."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    action_events: List[ExtractedActionEvent] = Field(default_factory=list)


def extraction_to_dict(result: ChunkExtraction, chunk_id: str, doc_id: str) -> dict:
    """Serialize a ChunkExtraction with grounding fields attached."""
    data = result.model_dump()
    data["chunk_id"] = chunk_id
    data["doc_id"] = doc_id
    return data


def extraction_from_dict(data: dict) -> ChunkExtraction:
    """Rebuild a ChunkExtraction from cached JSONL (ignores extra keys)."""
    payload = {k: data[k] for k in ("entities", "action_events") if k in data}
    return ChunkExtraction.model_validate(payload)


# ---------------------------------------------------------------------------
# Retrieval-side schemas (Pipeline 2)
# ---------------------------------------------------------------------------

class QueryEntity(BaseModel):
    """A named entity reference extracted from a user question."""

    name: str = Field(description="Canonical name of the entity the question depends on.")
    type: Optional[EntityType] = Field(
        default=None,
        description="Best-guess vertex type, if obvious; null otherwise.",
    )


class QueryEntities(BaseModel):
    """Gemini's structured output for query entity extraction."""

    entities: List[QueryEntity] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    """Final structured answer - validated JSON, not free text."""

    answer: str = Field(
        description="The direct answer (name/number/short phrase), or a statement "
        "of what is missing when evidence is insufficient."
    )
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="chunk_ids of the context blocks actually used.",
    )
    evidence_sufficient: bool = Field(
        description="False when the provided blocks do not contain the answer."
    )
    confidence: float = Field(description="Calibrated 0.0-1.0 confidence.")
    reasoning: str = Field(description="1-3 sentences naming which blocks support the answer.")


# ---------------------------------------------------------------------------
# Agentic layer schemas (Pipeline 3)
# ---------------------------------------------------------------------------

class AnswerSlot(BaseModel):
    """One independently answerable requirement carved out by the planner."""

    id: str = Field(description="Slot id, e.g. 's1'.")
    question: str = Field(description="The specific sub-question this slot asks.")
    slot_type: str = Field(
        default="relational",
        description="One of: relational | exact_fact | aggregation | overview.",
    )


class PlanOutput(BaseModel):
    """Planner output: decomposition into answer slots."""

    complexity: str = Field(description="'low' | 'medium' | 'high'.")
    answer_slots: List[AnswerSlot] = Field(
        default_factory=list,
        description="1 slot for simple questions; more only when independently answerable.",
    )


class SlotStatus(BaseModel):
    """Evidence verdict for one answer slot."""

    slot_id: str
    status: str = Field(description="'supported' | 'partial' | 'missing'.")
    support_score: float = Field(description="0.0-1.0 support strength.")
    evidence_chunk_ids: List[str] = Field(default_factory=list)
    note: str = Field(default="", description="One line: what is supported / what is missing.")


class EvidenceEvaluation(BaseModel):
    """Evidence evaluator output for one investigation round."""

    overall_status: str = Field(description="'sufficient' | 'partial' | 'insufficient'.")
    slots: List[SlotStatus] = Field(default_factory=list)
    contradiction: bool = Field(default=False)
    contradiction_note: str = Field(default="")
    ambiguity: bool = Field(default=False)
    missing: List[str] = Field(default_factory=list, description="Missing facts, e.g. 'exact medal time'.")
    recommended_action: str = Field(
        default="",
        description="Suggestion for the supervisor: GRAPH_SEARCH | VECTOR_SEARCH | COMMUNITY_SEARCH | STOP.",
    )


class SupervisorDecision(BaseModel):
    """Supervisor output: what to do next (also does the initial routing)."""

    action: str = Field(
        description="GRAPH_SEARCH | VECTOR_SEARCH | COMMUNITY_SEARCH | REFORMULATE | STOP"
    )
    target_slot: str = Field(default="", description="Slot id the action targets, '' if all/none.")
    subquery: str = Field(
        default="",
        description="The (possibly reformulated) query to execute; required unless STOP.",
    )
    reason_code: str = Field(
        description="e.g. initial_routing, missing_required_fact, wrong_relation_semantics, "
        "entity_not_found, contradiction, budget_exhausted, all_slots_supported."
    )
    note: str = Field(default="")
