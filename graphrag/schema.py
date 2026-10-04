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

import time

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


def _existing_vertex_attrs(conn: TigerGraphConnection, vtype: str) -> set[str]:
    """Attribute names declared on a vertex type in the live graph."""
    try:
        schema = conn.getSchema()
    except Exception:
        return set()
    for vt in schema.get("VertexTypes", []):
        if vt.get("Name") == vtype:
            return {a.get("AttributeName", a.get("name", "")) for a in vt.get("Attributes", [])}
    return set()


def ensure_context_attributes(conn: TigerGraphConnection, dry_run: bool = False) -> dict:
    """Add the v2 context attributes to ActionEvent via a SCHEMA_CHANGE job.

    Idempotent: introspects the live schema and alters only missing
    attributes. TG 4.x via the REST gsql endpoint requires the job form
    (the interactive `BEGIN SCHEMA_CHANGE` block fails to parse there):

        CREATE SCHEMA_CHANGE JOB <name> FOR GRAPH <g> {
          ALTER VERTEX ActionEvent ADD ATTRIBUTE (a STRING, b STRING);
        }
        RUN SCHEMA_CHANGE JOB <name>

    Waits until the live schema reports the attributes (the job applies
    asynchronously) before returning.

    Returns {"added": [...], "present": [...], "job": "..."}.
    """
    vtype = config.EVENT_VERTEX
    present = _existing_vertex_attrs(conn, vtype)
    missing = [a for a in config.CONTEXT_ATTRS if a not in present]
    have = [a for a in config.CONTEXT_ATTRS if a in present]

    if dry_run:
        return {"added": missing, "present": have, "job": "(dry-run)"}
    if not missing:
        print(f"Context attributes already declared on {vtype}.")
        return {"added": [], "present": have, "job": "none"}

    attr_sql = ", ".join(f"{n} {t}" for n, t in config.CONTEXT_ATTRS.items() if n in missing)
    job_name = f"add_context_attrs_{int(time.time())}"
    statement = (
        f"USE GRAPH {config.TG_GRAPHNAME}\n"
        f"CREATE SCHEMA_CHANGE JOB {job_name} FOR GRAPH {config.TG_GRAPHNAME} {{\n"
        f"ALTER VERTEX {vtype} ADD ATTRIBUTE ({attr_sql});\n"
        "}\n"
        f"RUN SCHEMA_CHANGE JOB {job_name}"
    )
    result = conn.gsql(statement)
    print(f"Schema change on {vtype}: added {missing}")

    # wait for the async job to reach the live schema (up to 120s)
    deadline = time.time() + 120
    still_missing = missing
    while time.time() < deadline:
        applied = _existing_vertex_attrs(conn, vtype)
        still_missing = [a for a in missing if a not in applied]
        if not still_missing:
            break
        time.sleep(3)
    if still_missing:
        raise RuntimeError(
            f"SCHEMA_CHANGE job did not apply within 120s; missing {still_missing}. "
            f"Job output: {str(result)[:300]}"
        )
    return {"added": missing, "present": have, "job": str(result)[:200]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print DDL without executing")
    args = parser.parse_args()

    connection = connect()
    verify_connection(connection)
    create_schema(connection, dry_run=args.dry_run)
