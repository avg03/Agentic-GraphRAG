"""GraphRAG configuration: TigerGraph connection + the graph ontology.

The ONTOLOGY below is the single source of truth for the whole pipeline:
  - shared.prompts embeds it into the extraction prompt (type list),
  - shared.models.EntityType constrains Gemini's enum,
  - graphrag.schema renders the GDDL from it,
  - graphrag.resolve / graphrag.load_graph walk it for dedup + upserts.

Env (loaded from .env):
    TIGERGRAPH_HOST     e.g. https://xxxx.i.tgcloud.io:443
    TIGERGRAPH_GRAPHNAME  e.g. GraphRAG (created by schema.py if missing)
    TIGERGRAPH_USERNAME default "tigergraph"
    TIGERGRAPH_PASSWORD required for TG Cloud (set during instance creation)
    TIGERGRAPH_SECRET   optional secret for creating a REST token
    TIGERGRAPH_TOKEN    optional pre-created REST token
    GEMINI_MODEL / GEMINI_THINKING_BUDGET / GOOGLE_API_KEY   (shared.llm_client)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# TigerGraph connection
# ---------------------------------------------------------------------------

TG_HOST = os.getenv("TIGERGRAPH_HOST") or os.getenv("TG_HOST")
TG_GRAPHNAME = (
    os.getenv("TIGERGRAPH_GRAPHNAME")
    or os.getenv("TIGERGRAPH_GRAPH")
    or os.getenv("TG_GRAPHNAME")
    or "GraphRAG"
)
TG_USERNAME = os.getenv("TIGERGRAPH_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TIGERGRAPH_PASSWORD", "")
TG_SECRET = os.getenv("TIGERGRAPH_SECRET") or os.getenv("TG_SECRET") or ""
TG_TOKEN = os.getenv("TIGERGRAPH_TOKEN", "")

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# Free-tier Gemini keys are limited to a few requests/minute per model —
# keep at 1 unless you're on a paid tier.
EXTRACTION_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "1"))
EXTRACTED_DIR = Path(__file__).resolve().parent / "extracted"

# ---------------------------------------------------------------------------
# Ontology — typed event-centric schema
# ---------------------------------------------------------------------------

# Non-event vertex types (+ the generic 'Entity' fallback bucket).
ENTITY_TYPES: list[str] = [
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

# The ActionEvent hub + the four contextual edge families.
EVENT_VERTEX = "ActionEvent"

# edge name -> allowed target vertex types (source is always ActionEvent)
EVENT_EDGES: dict[str, list[str]] = {
    "INITIATED_BY": ["Person", "Company", "Organization", "Team", "Country"],
    "TOOK_PLACE_AT": ["Location", "Venue", "Country"],
    "AFFECTED": [
        "Person",
        "Company",
        "Organization",
        "Team",
        "Country",
        "Location",
    ],
    "RESULTED_IN": ["Outcome"],
}

# Vertex attributes: type -> {attr: gsql_type} (PRIMARY_ID added by schema.py)
VERTEX_ATTRIBUTES: dict[str, dict[str, str]] = {
    "ActionEvent": {
        "name": "STRING",
        "action_type": "STRING",
        "description": "STRING",
        "impact_summary": "STRING",
    },
    "Chunk": {
        "doc_id": "STRING",
        "title": "STRING",
        "chunk_index": "INT",
        "url": "STRING",
    },
}
_COMMON_ENTITY_ATTRS = {"name": "STRING", "description": "STRING"}
for _t in ENTITY_TYPES:
    if _t == "Outcome":
        VERTEX_ATTRIBUTES[_t] = {**_COMMON_ENTITY_ATTRS, "outcome_kind": "STRING"}
    else:
        VERTEX_ATTRIBUTES[_t] = dict(_COMMON_ENTITY_ATTRS)

# Edge attributes
EDGE_ATTRIBUTES: dict[str, dict[str, str]] = {
    "MENTIONED_IN": {"source_chunk_id": "STRING"},
}

ALL_VERTEX_TYPES = [EVENT_VERTEX, *ENTITY_TYPES, "Chunk"]
ALL_EDGE_TYPES = [*EVENT_EDGES.keys(), "MENTIONED_IN"]


def validate_extraction_against_ontology(extraction: dict) -> tuple[dict, int]:
    """Drop extraction items violating the ontology; return (clean, dropped_count).

    Drops:
      - entities whose type is not in ENTITY_TYPES
      - event relation entries pointing at names that were not extracted as an
        allowed target type for that edge (force-map-or-drop decision)
    """
    dropped = 0
    name_to_type: dict[str, str] = {}
    entities = []
    for ent in extraction.get("entities", []):
        if ent.get("type") in ENTITY_TYPES and ent.get("name"):
            entities.append(ent)
            name_to_type.setdefault(ent["name"], ent["type"])
        else:
            dropped += 1

    events = []
    for event in extraction.get("action_events", []):
        if not event.get("name"):
            dropped += 1
            continue
        clean_event = {**event}
        for edge_name, allowed_types in EVENT_EDGES.items():
            kept = [
                n
                for n in event.get(_field_for_edge(edge_name), [])
                if name_to_type.get(n) in allowed_types
            ]
            dropped += len(event.get(_field_for_edge(edge_name), [])) - len(kept)
            clean_event[_field_for_edge(edge_name)] = kept
        events.append(clean_event)

    return {"entities": entities, "action_events": events}, dropped


def _field_for_edge(edge_name: str) -> str:
    """Map an edge name to its ExtractedActionEvent field."""
    return {
        "INITIATED_BY": "initiated_by",
        "TOOK_PLACE_AT": "took_place_at",
        "AFFECTED": "affected",
        "RESULTED_IN": "resulted_in",
    }[edge_name]
