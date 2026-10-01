"""GSQL installed-query traversal — BLOCKED by schema limitation (documented).

Status: NOT in use. `graphrag.traverse` uses the REST path
(`traverse_rest`), which is functionally superior here.

Why: the traversal needs to walk ActionEvent <- AFFECTED/RESULTED_IN -
(i.e. reverse edges of multi-endpoint directed edges). On this graph:

  1. TG 4.x schema change supports `ALTER EDGE ... ADD ATTRIBUTE/PAIR` only —
     `ADD REVERSE_EDGE` is not a valid alteration.
  2. `ADD EDGE <name> (...)` inside a schema-change job only accepts the name
     of an ALREADY-EXISTING global edge type, and global edge types cannot
     reference this graph's LOCAL vertex types.

So reverse_AFFECTED / reverse_RESULTED_IN cannot be created, and a GSQL query
can only traverse INITIATED_BY/TOOK_PLACE_AT events — silently missing
AFFECTED-mediated evidence (the richest family). REST `/edges/{type}/{id}`
returns both directions for every edge type, so `traverse_rest` covers the
full traversal correctly.

If the graph is ever recreated with global types (or reverse edges defined at
creation time, as INITIATED_BY/TOOK_PLACE_AT were), re-enable the installed
query: see QUERY_BODY_TEMPLATE below and `traverse.traverse_installed`.
"""

from graphrag.schema import connect

__all__ = ["QUERY_NAME", "QUERY_BODY_TEMPLATE"]

QUERY_NAME = "graphrag_traverse"

QUERY_BODY_TEMPLATE = """
CREATE QUERY {query_name}(SET<STRING> seeds) FOR GRAPH {graph} {{
  # Flat hits; Python aggregates support per chunk:
  #   direct_hits: "seed_entity_id|chunk_id"   event_hits: "event_id|chunk_id"
  # LIMITATION: event discovery below can only use reverse_INITIATED_BY /
  # reverse_TOOK_PLACE_AT (the only reverse edges that exist), so
  # AFFECTED/RESULTED_IN events are missed. REST traversal is authoritative.
  SetAccum<STRING> @@direct_hits;
  SetAccum<STRING> @@event_hits;

  AllTypes = {{Person.*, Company.*, Organization.*, Team.*, Country.*,
              Location.*, Venue.*, Sport.*, Outcome.*, Entity.*, ActionEvent.*}};
  SeedSet = SELECT v FROM AllTypes:v
            WHERE v.entity_id IN seeds OR v.event_id IN seeds;

  DirectChunks = SELECT c FROM SeedSet:s -(MENTIONED_IN:e)- Chunk:c
                 ACCUM @@direct_hits += (s.entity_id + "|" + c.chunk_id);

  SeedEvents = SELECT a FROM SeedSet:s
               -((reverse_INITIATED_BY|reverse_TOOK_PLACE_AT):e)- ActionEvent:a;

  EventChunks = SELECT c FROM SeedEvents:a -(MENTIONED_IN:e)- Chunk:c
                ACCUM @@event_hits += (a.event_id + "|" + c.chunk_id);

  PRINT @@direct_hits;
  PRINT @@event_hits;
}}
"""


def install_queries(conn=None) -> str:
    """Intentionally unavailable — see module docstring."""
    raise NotImplementedError(
        "GSQL traversal is blocked by missing reverse edges for AFFECTED/"
        "RESULTED_IN (see module docstring). graphrag.traverse uses the REST "
        "path, which covers all edge types in both directions."
    )


if __name__ == "__main__":
    print(__doc__)
