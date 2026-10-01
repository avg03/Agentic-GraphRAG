"""TigerGraph connection + idempotent schema creation from the config ontology.

Renders GSQL DDL from `graphrag.config` (single source of truth) and creates
the vertex/edge types and the graph itself. Safe to re-run: existing types
are detected and skipped.

Usage:
    from graphrag.schema import connect, create_schema, verify_connection

    conn = connect()
    verify_connection(conn)
    create_schema(conn)   # creates types + graph if missing

Env: TIGERGRAPH_HOST, TIGERGRAPH_GRAPHNAME, TIGERGRAPH_USERNAME,
     TIGERGRAPH_PASSWORD, TIGERGRAPH_SECRET / TIGERGRAPH_TOKEN.
"""

from pyTigerGraph import TigerGraphConnection

from graphrag import config

__all__ = ["connect", "verify_connection", "build_ddl", "create_schema"]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect() -> TigerGraphConnection:
    """Open a TigerGraphConnection from env config (auth via gsqlSecret + token).

    The `gsqlSecret=` parameter is what makes GSQL-statement auth work on
    TG Cloud (plain username/password without a password fails with
    "User authentication failed"); the token from the secret authorizes
    the REST++ upsert endpoints.
    """
    if not config.TG_HOST:
        raise RuntimeError(
            "TIGERGRAPH_HOST is not set. Add TIGERGRAPH_HOST / TIGERGRAPH_SECRET "
            "(and optionally TIGERGRAPH_TOKEN) to .env — values come from your "
            "TigerGraph Cloud instance page."
        )

    conn = TigerGraphConnection(
        host=config.TG_HOST,
        graphname=config.TG_GRAPHNAME,
        username=config.TG_USERNAME,
        password=config.TG_PASSWORD,
        gsqlSecret=config.TG_SECRET or None,
    )

    if config.TG_TOKEN:
        conn.apiToken = config.TG_TOKEN
    elif config.TG_SECRET:
        try:
            conn.getToken(config.TG_SECRET)
        except Exception as exc:
            raise RuntimeError(
                f"Could not obtain a REST token from TIGERGRAPH_SECRET: {exc}"
            ) from exc

    return conn


def verify_connection(conn: TigerGraphConnection) -> str:
    """Ping the server; raises if unreachable."""
    response = conn.echo(usePost=True)
    print(f"Connected to {config.TG_HOST} — echo: {response}")
    return response


# ---------------------------------------------------------------------------
# DDL rendering (from config ontology)
# ---------------------------------------------------------------------------

def _render_vertex_type(vtype: str) -> str:
    attrs = config.VERTEX_ATTRIBUTES[vtype]
    if vtype == "ActionEvent":
        primary = "PRIMARY_ID event_id STRING"
    elif vtype == "Chunk":
        primary = "PRIMARY_ID chunk_id STRING"
    else:
        primary = "PRIMARY_ID entity_id STRING"
    attr_sql = ", ".join(f"{name} {gtype}" for name, gtype in attrs.items())
    stats = 'STATS="outdegree_by_edgetype"' if vtype != "Chunk" else 'STATS="none"'
    return f"CREATE VERTEX {vtype} ({primary}, {attr_sql}) WITH {stats}"


def _render_edge_type(ename: str) -> str:
    if ename == "MENTIONED_IN":
        # Every entity/event type may be mentioned in a chunk.
        sources = ", ".join(f"FROM {v}" for v in config.ALL_VERTEX_TYPES if v != "Chunk")
        endpoints = f"{sources}, TO Chunk"
    else:
        targets = " | TO ".join(config.EVENT_EDGES[ename])
        endpoints = f"FROM {config.EVENT_VERTEX}, TO {targets}"
    attrs = config.EDGE_ATTRIBUTES.get(ename, {})
    attr_sql = "".join(f", {name} {gtype}" for name, gtype in attrs.items())
    return f"CREATE DIRECTED EDGE {ename} ({endpoints}{attr_sql})"


def build_ddl() -> dict:
    """Render the DDL statements (global types + graph creation) as a list."""
    statements = ["USE GLOBAL"]
    statements += [_render_vertex_type(v) for v in config.ALL_VERTEX_TYPES]
    statements += [_render_edge_type(e) for e in config.ALL_EDGE_TYPES]
    statements.append(
        f"CREATE GRAPH {config.TG_GRAPHNAME} ({', '.join(config.ALL_VERTEX_TYPES + config.ALL_EDGE_TYPES)})"
    )
    return statements


# ---------------------------------------------------------------------------
# Execution (idempotent)
# ---------------------------------------------------------------------------

_ALREADY_EXISTS_MARKERS = ("already exists", "is already", "exists in the catalog")


def _run(conn: TigerGraphConnection, statement: str) -> str:
    result = conn.gsql(statement)
    print(f"  {statement.splitlines()[-1][:100]} -> OK")
    return result


def _existing_graph_types(conn: TigerGraphConnection) -> tuple[set, set] | None:
    """Vertex/edge type names already in the target graph; None if graph absent."""
    try:
        schema = conn.getSchema()
    except Exception:
        return None
    return (
        {vt["Name"] for vt in schema.get("VertexTypes", [])},
        {et["Name"] for et in schema.get("EdgeTypes", [])},
    )


def create_schema(conn: TigerGraphConnection, dry_run: bool = False) -> dict:
    """Ensure the graph carries the full ontology. Idempotent.

    Three cases:
      1. Graph already has every type (e.g. created via GraphStudio) — skip
         DDL entirely. Creating global types on an instance that holds a
         same-named type (e.g. a dataset's `Country`) fails with a semantic
         check error.
      2. Graph absent — create global types + graph via build_ddl().
      3. Graph present but incomplete — raise with the missing types listed
         (extending a live graph is left to a deliberate schema-change job).
    """
    if dry_run:
        for s in build_ddl():
            print(s)
        return {"created": [], "skipped": [], "graph": config.TG_GRAPHNAME}

    existing = _existing_graph_types(conn)
    created, skipped = [], []

    if existing is not None:
        have_v, have_e = existing
        missing_v = [v for v in config.ALL_VERTEX_TYPES if v not in have_v]
        missing_e = [e for e in config.ALL_EDGE_TYPES if e not in have_e]
        if not missing_v and not missing_e:
            print(
                f"Schema ready: graph {config.TG_GRAPHNAME!r} already carries the "
                f"full ontology ({len(have_v)} vertex types, {len(have_e)} edge types)."
            )
            conn.graphname = config.TG_GRAPHNAME
            return {"created": [], "skipped": sorted(have_v | have_e), "graph": config.TG_GRAPHNAME}
        raise RuntimeError(
            f"Graph {config.TG_GRAPHNAME!r} exists but is incomplete. "
            f"Missing vertex types: {missing_v}; edge types: {missing_e}. "
            "Extend it with a SCHEMA_CHANGE job (see graphrag/schema.py build_ddl) "
            "or point TIGERGRAPH_GRAPHNAME at a fresh graph."
        )

    # Fresh graph: run global DDL then create the graph.
    for statement in build_ddl():
        if statement == "USE GLOBAL":
            continue
        label = statement.split("(")[0].strip()
        try:
            _run(conn, statement)
            created.append(label)
        except Exception as exc:
            msg = str(exc).lower()
            if any(marker in msg for marker in _ALREADY_EXISTS_MARKERS) or "used by another object" in msg:
                print(f"  {label} -> already exists, skipping")
                skipped.append(label)
            else:
                raise

    conn.graphname = config.TG_GRAPHNAME
    print(f"Schema ready: {len(created)} created, {len(skipped)} already present.")
    return {"created": created, "skipped": skipped, "graph": config.TG_GRAPHNAME}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print DDL without executing")
    args = parser.parse_args()

    connection = connect()
    verify_connection(connection)
    create_schema(connection, dry_run=args.dry_run)
