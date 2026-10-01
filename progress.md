# GraphRAG Pipeline — Progress Log

> **Companion to `context.md`.** This file records what has been built, how it
> was built, every error encountered and how it was resolved, and what remains.
> Update it after each milestone.

**Last updated:** 2026-09-25

---

## 1. Current Status Snapshot

| Component | Status |
|---|---|
| Pipeline 1 (Vector RAG: BM25 + Dense + Reranker) | ✅ Done (pre-existing) |
| One-time corpus ingestion → ChromaDB | ✅ **Done** — 150 docs → 1,347 chunks in ChromaDB |
| `shared/` package (LLM client, prompts, metrics, models) | ✅ **Done** |
| GraphRAG extraction (entity + relation builder) | ✅ **Done** — pilot complete: 112/112 chunks extracted (gemini-flash-lite-latest) |
| Graph schema (TigerGraph DDL) | ✅ **Live** — graph `Transaction_Fraud` carries the full ontology; MENTIONED_IN covers all entity types |
| Dedup / resolve (graph data prep) | ✅ Done — verified on cached extractions |
| TigerGraph load (Stage 4) | ✅ **Done & verified** — pilot: 961 vertices / 2,177 edges live in TG Cloud; counts + traversal + grounding all verified |
| Graph traversal retrieval | ✅ Code done — **smoke-tested** (extraction+rank+answer verified; graph stages pending instance restart) |
| Prompt compilation + final answer LLM call | ✅ Code done — anti-hallucination system prompt + Pydantic `GroundedAnswer`; verified live |
| Eval harness / metrics comparison | ✅ `graphrag/eval_run.py` written (`--pilot` mode); full run after corpus scale-up |
| Pipeline 3 (Agentic GraphRAG) | ❌ Not started |

---

## 2. What Was Built

### 2.1 `shared/` — reusable layer (per context.md spec; did not exist before)

| File | Contents |
|---|---|
| `shared/__init__.py` | Package doc |
| `shared/llm_client.py` | Gemini wrapper on **google-genai** SDK (v2.24.0). Key functions: `get_client()` (multi-key pool), `generate_structured(prompt, response_schema, system, ...)` → `(parsed_pydantic, usage_dict)` with retry/backoff + key rotation; `generate_text(...)` for plain completions. Reads `GEMINI_MODEL` (default `gemini-3.8-flash`), `GEMINI_THINKING_BUDGET` (default 0 = thinking off, ~7s vs ~30-45s per call). |
| `shared/models.py` | Pydantic models doubling as Gemini `response_schema` and cache parse targets: `ExtractedEntity` (name, type enum, description), `ExtractedActionEvent` (name, action_type, description, impact_summary, initiated_by[], took_place_at[], affected[], resulted_in[]), `ChunkExtraction` (entities[], action_events[]). The entity `type` is a **hard Literal enum** in the response schema so Gemini cannot return off-ontology types. Helpers: `extraction_to_dict` / `extraction_from_dict`. |
| `shared/prompts.py` | `EXTRACTION_SYSTEM_PROMPT` (8 rules: only-what's-stated, ActionEvent hub, exact relation names, canonical naming for cross-chunk resolution, empty-lists-if-nothing) and `build_extraction_prompt(title, chunk_text, entity_types)`. |
| `shared/metrics.py` | `Metrics` class: `incr(counter)`, `timer(name)` context manager, `record_llm(usage, latency_ms)`, `summary()` / `pretty()`. Same metric shape for all 3 pipelines (token counts from Gemini `usage_metadata`, latency, arbitrary counters). |

### 2.2 `graphrag/` — GraphRAG ingestion pipeline (all new; folder was empty)

| File | Contents |
|---|---|
| `graphrag/config.py` | **Single source of truth for the ontology** (see §4). Also: TG env vars (`TIGERGRAPH_HOST/GRAPHNAME/USERNAME/PASSWORD/SECRET/TOKEN`), `EXTRACTION_CONCURRENCY` (default 1 for free tier), `EXTRACTED_DIR` cache path, and `validate_extraction_against_ontology(extraction)` → drops off-ontology entities/relations ("force-map-or-drop") and returns a dropped-count. |
| `graphrag/schema.py` | `connect()` (pyTigerGraph `TigerGraphConnection`, auto-requests REST token from secret), `verify_connection(conn)` (echo ping), `build_ddl()` (renders GSQL from config ontology), `create_schema(conn)` (**idempotent** — skips types that already exist). CLI: `python -m graphrag.schema --dry-run`. |
| `graphrag/extract.py` | Stage 2. `extract_chunks(chunks, concurrency, metrics)` — per-chunk Gemini structured extraction; **checkpoint cache** `graphrag/extracted/{doc_id}.jsonl` (one line per chunk) so re-runs skip already-extracted chunks; ThreadPoolExecutor (concurrency 1 on free tier); validates each result against the ontology and counts drops. Also `load_cached_extractions(docs=None)`. |
| `graphrag/resolve.py` | Stage 3 — **deterministic dedup, zero LLM cost**. `normalize_name()` (lowercase, strip punctuation, drop leading the/a/an, collapse whitespace); `resolve_extractions(records)` → merges entity mentions by (type, normalized name), picks most-frequent original casing as display name, merges ActionEvents by normalized name (unions relations + descriptions), retargets relations through the alias map, and emits `MENTIONED_IN` edges from every vertex to every chunk it appeared in. Output: `{"vertices": {vtype: {vid: {attributes, mentioned_in}}}, "edges": [{edge, from_type, from_id, to_type, to_id}], "aliases"}`. |
| `graphrag/load_graph.py` | Stage 4. `resolved_to_upserts()` converts resolve output into `upsertData` payloads (`{"vertices": ..., "edges": ...}` REST format) in batches of 2000 ops; also adds `Chunk` vertices for every referenced chunk. `load_resolved(conn, resolved, chunk_records)` does the batched `conn.upsertData()` calls. `load_chunks(conn, records)` for standalone Chunk vertices. MENTIONED_IN edges carry `source_chunk_id` for provenance. |
| `graphrag/ingest_pipeline.py` | **Driver**: `run_pipeline(doc_ids, skip_load, metrics)` = Stage 1 (read chunks back from Chroma via `collection.get` — **never re-chunks**; ingestion runs once via `ingestion/run_ingest.py`) → Stage 2 extract (cached) → Stage 3 resolve (prints per-type vertex counts) → Stage 4 TigerGraph load (skippable). CLI flags: `--n-docs N` / `--full` / `--doc-ids` / `--skip-load`. Prints metrics summary at the end. |

### 2.3 `ingestion/run_ingest.py` — one-time vector ingestion driver (new)

- `ingest_corpus(corpus_path, n_docs, reset, collection, batch_size)`:
  `load_corpus()` → `chunk_documents()` → `embed_chunks()` → `collection.add()`
  in batches of 128. **Idempotent**: skips chunk_ids already in the collection.
  Sanitizes metadata (Chroma rejects `None` values — `approx_tokens` is often None).
- CLI: `--n-docs N` (pilot) / `--full`, `--reset`, `--batch-size`.
- **Run result:** 150 docs → **1,347 chunks** (512 tokens / 54 overlap, cl100k_base) in
  collection `tigergraph_docs`.

### 2.4 Existing components reused (built earlier, unchanged)

| Component | File | Role in GraphRAG pipeline |
|---|---|---|
| Corpus loader | `ingestion/preprocessing.py` → `load_corpus()` | JSONL → LangChain Documents (dedup by doc_id, clean_text) |
| Chunker | `ingestion/chunking.py` → `chunk_documents()` | 512-token chunks, **stable IDs** `{doc_id}_chunk_{index}` = join key for Chunk vertices + MENTIONED_IN |
| Embedder | `ingestion/store_embed.py` → `embed_chunks()` | BGE bge-small-en-v1.5 (384-dim, normalized) |
| Chroma client | `database/chroma_client.py` | Persistent local DB, `get_collection()` / `reset_collection()` |
| Pipeline-1 retrieval | `retrieval/*` (dense, bm25, dedup, rerank, main_pipeline) | Baseline pipeline; its `load_bm25_corpus_from_chroma` pattern reused for reading chunks back |
| Eval data | `Data/eval_public.jsonl` (100 q), `Data/eval_hidden.jsonl` (50 q) | Benchmark questions with gold doc_ids |

---

## 3. Graph Schema (as implemented)

### Vertices (10 typed + generic fallback + Chunk)

```gsql
CREATE VERTEX ActionEvent (PRIMARY_ID event_id STRING, name STRING, action_type STRING,
                           description STRING, impact_summary STRING) WITH STATS="outdegree_by_edgetype";
CREATE VERTEX Person       (PRIMARY_ID entity_id STRING, name STRING, description STRING) ...;
CREATE VERTEX Company      (...)  CREATE VERTEX Organization (...)
CREATE VERTEX Team         (...)  CREATE VERTEX Country      (...)
CREATE VERTEX Location     (...)  CREATE VERTEX Venue        (...)
CREATE VERTEX Sport        (...)
CREATE VERTEX Outcome      (PRIMARY_ID entity_id STRING, name STRING, description STRING, outcome_kind STRING);
CREATE VERTEX Entity       (PRIMARY_ID entity_id STRING, name STRING, description STRING);  -- fallback bucket
CREATE VERTEX Chunk        (PRIMARY_ID chunk_id STRING, doc_id STRING, title STRING, chunk_index INT, url STRING);
```

Vertex IDs are the **normalized names** (unique within a vertex type, which is
all TigerGraph requires of a PKEY). Display name kept in the `name` attribute.

### Edges (user-specified event-centric hub + grounding edge)

```gsql
CREATE DIRECTED EDGE INITIATED_BY  (FROM ActionEvent, TO Person | TO Company | TO Organization | TO Team | TO Country);
CREATE DIRECTED EDGE TOOK_PLACE_AT (FROM ActionEvent, TO Location | TO Venue | TO Country);
CREATE DIRECTED EDGE AFFECTED      (FROM ActionEvent, TO Person | TO Company | TO Organization | TO Team | TO Country | TO Location);
CREATE DIRECTED EDGE RESULTED_IN   (FROM ActionEvent, TO Outcome);
CREATE DIRECTED EDGE MENTIONED_IN  (FROM ActionEvent, FROM Person, ..., FROM Entity, TO Chunk, source_chunk_id STRING);  -- grounding
```

Design decisions (user-approved):
- **Typed core + generic `Entity` fallback** — anything not fitting a type goes
  to `Entity`; nothing is lost.
- **Force-map-or-drop** — relations that don't map onto the 4 edge families are
  dropped and counted (`ontology_dropped` counter) to see if the ontology needs
  widening after the pilot.
- **MENTIONED_IN kept for ALL vertex types** (per context.md grounding rule);
  carries `source_chunk_id`.
- Types added beyond the user's original 7 for the Wikipedia/Olympics corpus:
  `Organization`, `Team`, `Sport`, `Outcome`.

### Data flow

```
Data/corpus.jsonl
  → load_corpus() → chunk_documents()          [ONCE, ingestion/run_ingest.py]
  → embed_chunks() → ChromaDB (1,347 chunks)   [shared by ALL pipelines]
       ↓ collection.get (by doc_id)
  → Gemini structured extraction per chunk     [graphrag/extract.py, cached JSONL]
  → validate against ontology (drop+count)     [graphrag/config.py]
  → dedup/resolve (normalize, merge, alias)    [graphrag/resolve.py, no LLM]
  → batched upsertData to TigerGraph           [graphrag/load_graph.py]
       (+ Chunk vertices; MENTIONED_IN edges to chunk IDs)
```

---

## 4. Errors Encountered & Resolutions

| # | Error | Root cause | Fix / Resolution |
|---|---|---|---|
| 1 | `404 NOT_FOUND: model gemini-2.5-flash is no longer available to new users` | Deprecated model on a new API key (Sep 2026) | Default `GEMINI_MODEL` switched to **`gemini-3.8-flash`** (API's own suggestion). Model list fetched via `client.models.list()` to confirm alternatives. |
| 2 | `429 RESOURCE_EXHAUSTED ... limit: 5` (per-minute) | Free tier = 5 requests/min for gemini-3.8-flash | `_retry_delay()` now **parses the server's "retry in Ns" hint** from the error and sleeps exactly that long (+2s, capped 120s); `EXTRACTION_CONCURRENCY` default set to 1. |
| 3 | `UnboundLocalError: cannot access local variable '_client_pool'` — **killed 109/112 pilot chunks** | Key-rotation function `_rotate_client()` assigned `_client_pool` but omitted it from its `global` declaration → first quota error crashed every call | Added `_client_pool` to the `global` statement; verified with live call. Checkpointing meant only the 3 already-cached chunks were preserved — rerun resumed for free. |
| 4 | `429 ... GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20` | **Free-tier DAILY cap for gemini-3.8-flash is only 20 requests/day/project** — the buggy run drained it (each failed chunk still consumed one attempt before crashing) | Pilot extraction paused. Verified `gemini-flash-lite-latest` and `gemini-3.5-flash-lite` respond fine (separate per-model quota pools, much higher free limits). **Pending decision: switch `GEMINI_MODEL=gemini-flash-lite-latest` and resume.** |
| 5 | Chroma `ValueError` risk on metadata | `approx_tokens` can be `None`; Chroma rejects None metadata values | `_sanitize_metadata()` in `run_ingest.py` keeps only str/int/float/bool values. |
| 6 | MENTIONED_IN edges all attributed to ActionEvent | Multi-source edge type — source vertex type must be stated per edge instance | `resolve_extractions()` now emits `from_type` on every MENTIONED_IN edge; `load_graph._source_vtype()` reads it. |
| 7 | (Earlier, noted in chroma_client.py) `import chromadb` shadowing | A file named `chromadb.py` would shadow the real package | File deliberately named `chroma_client.py` (pre-existing fix, documented). |
| 8 | `User authentication failed` on getToken + GSQL | TG Cloud auth needs the `gsqlSecret=` connection param — plain username/password (with empty password) is rejected | **User-provided approach adopted**: `TigerGraphConnection(host, graphname, gsqlSecret=secret)` then `conn.getToken(secret)`. Both GSQL and REST auth now work. Env aliases `TIGERGRAPH_GRAPH` / `TG_GRAPHNAME` also accepted. |
| 9 | `REST-10004: Unknown vertex attribute "attributes"` on upsert | TG 4.x upsert wire format puts attributes DIRECTLY under the vertex ID as `{"attr": {"value": v}}` — no `"attributes"` wrapper (pyTigerGraph's `upsertData` docs describe the 3.x shape) | `graphrag/load_graph.py` rewritten to emit `{"vertices": {type: {id: {attr: {"value": v}}}}, "edges": {from_type: {from_id: {EDGE: {to_type: {to_id: {}}}}}}}`. Verified: `accepted_vertices: 1`. |
| 10 | `REST-30200: value cannot be converted to STRING` for chunk_index | The live graph declares `Chunk.chunk_index` as STRING (not INT as config assumed) | `load_graph.py` sends `chunk_index` as string. (Also noted: `Team` pkey is `entity_d` — a typo in the manually-created graph, harmless since upserts are pkey-name-agnostic.) |
| 11 | `REST-30200: from vertex type 'Person' ... not valid for edge type 'MENTIONED_IN'` | The manually-created graph's MENTIONED_IN only had pairs `Entity|ActionEvent → Chunk`; grounding needs every entity type | Ran a **SCHEMA_CHANGE job** (`ALTER EDGE MENTIONED_IN ADD PAIR (FROM X, TO Chunk)` × 9 types) — graph v4 → v5. Note pyTigerGraph's `gsql(graphname=...)` arg is a no-op; `USE GRAPH` must be inside the statement text. |
| 12 | Vertex counts read 0 right after upsert | RESTPP indexing lag | Counts appear after a few seconds (99/99 confirmed). |
| 13 | `_get(.../edges/...)` returns a flat list of edge objects (36 items), not `{"results": [...]}` pages | pyTigerGraph `_get` unwraps multi-page responses | Iterate the list directly; dedupe by (edge, to_type, to_id). |
| 14 | `Semantic Check Fails: The vertex name Country is used by another object!` during `create_schema` | The crashed-run fix attempt tried GLOBAL DDL; the instance's fraud dataset already has a global `Country` type | `create_schema()` now introspects `getSchema()` first: graph-complete → skip DDL; graph absent → global DDL; graph incomplete → raise with guidance. |
| 15 | Stray global vertex types (ActionEvent, Person, Company, Organization, Team) created by the crashed run | `create_schema` ran global `CREATE VERTEX` before hitting the Country conflict | Dropped via `USE GLOBAL\nDROP VERTEX <type>` (TG 4.x syntax — no GLOBAL keyword on DROP). Instance restored to pre-run state. |
| 16 | Partial upsert accept (543/961 vertices) on first pilot load | Name collision between stray global types and graph-local types confused RESTPP | After dropping the stray globals, the same payload accepted 961/961 vertices + 2,177/2,177 edges. |

**Note on an earlier wrong estimate:** it was initially claimed the buggy run
burned "~600 requests" (109 failures × 6 retries). Reconstructed from logs: the
UnboundLocalError **short-circuited retries** — each failed chunk made ~1
request, so ~110 attempts, most rejected with 429. The daily cap died because
it is only 20/day, not because of retry volume.

---

## 5. Verification Evidence

- **DDL rendering** (no server needed): `build_ddl()` output inspected — all 12
  vertex types, 5 edge types, `CREATE GRAPH GraphRAG (...)` correct.
- **Dedup logic** (synthetic test): "United States" + "the United States" merged
  into one `Country:united states` vertex with MENTIONED_IN to both chunks;
  "Paris" (Location) correctly dropped from TOOK_PLACE_AT (Location not allowed
  there — force-map-or-drop working).
- **Live extraction** (real corpus chunks, gemini-3.8-flash): e.g.
  *"2012 Summer Olympics Men's K-2 1000 metres"* → Venue Eton Dorney, Location
  London, 8 medalist Persons, 3 Countries, Gold/Silver/Bronze Outcomes, clean
  event hub. ~27s/chunk with thinking on, ~7s with thinking off.
- **Resolve on cached data** (7 chunks): 3 ActionEvents, 87 entity vertices
  (61 Person, 20 Country, 2 Location, 2 Venue, 4 Outcome), 94 event edges,
  126 MENTIONED_IN edges — extrapolates to a substantial graph at full scale.
- **Multi-key pool**: both `.env` keys detected, both independently
  authenticate (tested on gemini-flash-lite-latest).
- **ChromaDB**: `collection.count() == 1347`.

---

## 6. How to Run (so far)

```bash
# 0. venv: D:\TigerGraph\ragvenv (Python 3.14.2)

# 1. One-time vector ingestion (shared by all pipelines)
./ragvenv/Scripts/python.exe -m ingestion.run_ingest --n-docs 150   # pilot
./ragvenv/Scripts/python.exe -m ingestion.run_ingest --full         # full corpus

# 2. GraphRAG ingestion (extraction → resolve → [load])
./ragvenv/Scripts/python.exe -m graphrag.ingest_pipeline --n-docs 20 --skip-load
./ragvenv/Scripts/python.exe -m graphrag.ingest_pipeline --n-docs 150          # + TG load
./ragvenv/Scripts/python.exe -m graphrag.schema --dry-run                      # preview DDL

# 3. TigerGraph schema (after .env has TIGERGRAPH_HOST/PASSWORD/SECRET)
./ragvenv/Scripts/python.exe -m graphrag.schema
```

### Environment variables (.env)

| Var | Status | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ set | Gemini key #1 |
| `GOOGLE_API_KEY_2` | ✅ set | Gemini key #2 (quota fallback; different account — separate quota pool) |
| `GEMINI_MODEL` | unset → `gemini-3.8-flash` | **Pending: set to `gemini-flash-lite-latest`** (3.8-flash daily cap = 20 req/day, exhausted) |
| `GEMINI_API_KEYS` | unset | Optional comma-separated key pool (alternative to GOOGLE_API_KEY_2) |
| `GEMINI_THINKING_BUDGET` | unset → 0 | Thinking tokens (0 = off) |
| `TIGERGRAPH_HOST` | ✅ set | TG Cloud instance URL |
| `TIGERGRAPH_GRAPHNAME` | ✅ set = `Transaction_Fraud` | Graph (pre-exists with our ontology; `TIGERGRAPH_GRAPH`/`TG_GRAPHNAME` also accepted) |
| `TIGERGRAPH_SECRET` | ✅ set | Used via `gsqlSecret=` param + `getToken()` |
| `TIGERGRAPH_PASSWORD` | not needed | gsqlSecret path works without it |
| `TIGERGRAPH_TOKEN` | optional | Pre-created REST token (we mint one from the secret instead) |

### TigerGraph live instance notes

- Graph `Transaction_Fraud` was created manually (GraphStudio) with our exact
  ontology — local types, correct PKEYs and attributes; the instance also
  carries leftover global fraud-dataset types (Phone, Email, Account, ...),
  unused by our graph.
- `Chunk.chunk_index` is STRING there; `Team` pkey is `entity_d` (typo, harmless).
- `MENTIONED_IN` was extended by schema-change job to cover all 11 entity/event
  types → Chunk (graph version 5).
- Verified end-to-end on 7 cached chunks: **99 vertices / 220 edges upserted**
  (3 ActionEvent, 61 Person, 20 Country, 2 Location, 2 Venue, 4 Outcome, 7 Chunk);
  traversal from the speed-skating event shows TOOK_PLACE_AT/AFFECTED/RESULTED_IN/
  MENTIONED_IN edges and chunk IDs matching Chroma exactly.

### PILOT RESULT (20 docs, complete pipeline run)

- Extraction: **112/112 chunks** via `gemini-flash-lite-latest` (~3-4s/chunk,
  no quota stalls; checkpointed in `graphrag/extracted/`).
- Resolved graph: 61 ActionEvent, 584 Person, 85 Country, 39 Sport, 19 Venue,
  31 Outcome, 13 Location, 17 Entity(fallback), 3 Organization, 3 Team.
- Loaded into TigerGraph: **961 vertices / 2,177 edges** (691 event edges +
  1,486 MENTIONED_IN grounding edges); server accepted 100% after the
  stray-global cleanup; live `getVertexCount` matches resolve exactly.
- Grounding verified: e.g. Person "Sven Kramer" → MENTIONED_IN → Q607635_chunk_0/1
  (the same IDs as ChromaDB).

---

## 8. Pipeline 2 — Retrieval & Answer (built 2026-09-25)

Flow: question → Gemini query-entity extraction → entity match (normalized +
BGE sim ≥ 0.75) → TigerGraph traversal → Chroma chunk lookup → support-sort +
BGE rerank + sufficiency gate (dense top-up, chunks tagged graph|dense) →
anti-hallucination prompt → structured `GroundedAnswer` (answer, cited_chunk_ids,
evidence_sufficient, confidence, reasoning).

New modules: `graphrag/entity_index.py` (vertex catalog + embeddings, cached
`entity_index.json/.npz`), `graphrag/traverse.py` (REST primary, installed-GSQL
primary path scaffolded in `graphrag/install_queries.py`), `graphrag/chunk_lookup.py`,
`graphrag/rank.py`, `graphrag/answer.py`, `graphrag/retrieval_pipeline.py`
(driver: `answer_question()`), `graphrag/eval_run.py`.
Shared extensions: `QueryEntities` / `GroundedAnswer` schemas, query-entity +
answer prompts in `shared/`.

Offline smoke test (graph hibernating): correct answer "Rudolf Dombi and Roland
Kökény" for the K-2 1000m question, cited to the graph chunk, gate triggered
dense top-up correctly. Blocked on TG instance restart: entity index build,
traverse test, install_queries test, full pilot eval.

### Pipeline 2 live verification (instance resumed)

- Entity index built (855 vertices cached); matching verified: "Sven Kramer"
  exact, "speed skating 5000m at 2010 olympics" → ActionEvent via sim=0.90.
- REST traversal verified: Sven Kramer + South Korea seeds → 5 chunks with
  correct direct/event support attribution.
- End-to-end driver: "Who won the gold medal in the men's 5000 metres speed
  skating at the 2010 Winter Olympics?" → **"Sven Kramer"**, cited
  Q607635_chunk_0, 4 graph chunks, zero dense top-up, 2 LLM calls / 2,670 tokens.
- GSQL installed query BLOCKED (documented in `graphrag/install_queries.py`):
  TG 4.x schema change can't add reverse edges to multi-endpoint edges
  (ALTER supports ATTRIBUTE/PAIR only; ADD EDGE can't reference local vertex
  types), so a GSQL traversal would silently miss AFFECTED/RESULTED_IN events.
  REST traversal (returns both directions for all edge types) is authoritative.
  Leftover draft query dropped; stray jobs cleaned.

### Pilot eval (6 questions touching the 20 pilot docs)

- 0/6 correct — and correctly so: these eval questions (aggregation/superlative)
  require evidence across 11-22 gold docs; pilot graph covers 5-10% of them.
  Every answer was an honest `evidence_sufficient=false` refusal — the
  anti-hallucination rules working as designed, not retrieval failures.
- Scoring bug found+fixed: substring match made gold "20" match "2004";
  matcher now uses word boundaries (substring only for golds > 3 chars).
- Meaningful accuracy numbers require the full-corpus graph (next step).

## 9. Next Steps (updated)

1. **Full-corpus ingestion** — `ingestion.run_ingest --full` (~2,951 docs →
   ~13k chunks), then extraction at lite-model speed with raised concurrency
   (test lite-tier RPM limits first), entity index rebuilds automatically.
2. **Full eval + Pipeline 1 comparison** — `graphrag.eval_run` (all 100
   public questions) vs `retrieval.main_pipeline` + same answer stage;
   identical metrics for the README comparison table.
3. **Pipeline 3 (Agentic GraphRAG)** — planner/routing/evidence loop on top
   of Pipelines 1 & 2 retrievers (consumes `GroundedAnswer` /
   `evidence_sufficient` programmatically).

## 10. Pipeline 3 — Agentic GraphRAG (built 2026-09-26)

LangGraph-based investigation layer over Pipelines 1 & 2 (`pip install langgraph`).
Flow: planner (Gemini structured decomposition into answer slots) → **merged
Supervisor** (routes first action + every next action; budget enforced before
the LLM call) → tools → evaluator (per-slot support matrix, contradiction +
ambiguity flags, semantic validation of graph paths) → conditional loop →
context builder (dedupe, graph paths + numbered chunks) → grounded
`GroundedAnswer`.

New package `agentic/`: config (MAX_STEPS=5, MAX_GRAPH_CALLS=3, MAX_VECTOR_CALLS=2,
MAX_COMMUNITY_CALLS=1, env-driven), state (AgentState TypedDict + RAGResult §18),
planner, tools (graph/vector/community), evaluator, supervisor, context_builder,
answer, graph (StateGraph wiring), agent (`run_agentic`), eval_run.
Extensions: agentic schemas + prompts in `shared/`; `traverse.collect_relationships()`
(REST edge endpoints → path strings, event-hub-first, capped).

Community Search = lightweight doc-level proxy: BM25 chunks grouped by doc_id
into "communities" (honest early-stage version; real community detection later).

### Live verification (pilot graph + full-corpus Chroma)

1. **Demo question** (doc §22): "Which athletes from Hungary won medals in the
   2012 men's K-2 1000m event, where was it held, and what were their times?"
   → 3 slots → GRAPH_SEARCH → all slots supported → **"Rudolf Dombi and Roland
   Kökény … Eton Dorney in London … 3:09.646"**, cited, 7,325 tokens.
2. **Strategy switch demo**: "Who won gold in men's table tennis 2008?" →
   graph miss (entity_not_found) → supervisor switched to VECTOR_SEARCH →
   **"Ma Lin"**, `strategy_changed=true`. (Graph lacked that doc — extraction
   still in progress — vector search over full Chroma found it.)
3. **Ambiguity/budget**: vague Paris question → ambiguity flagged, 5-step
   budget respected, tools varied, honest refusal (corpus lacks Paris 2024).

### Agentic pilot eval (10 questions touching ingested docs)

- **Accuracy 40%** (4/10 — all temporal/lookup/multi_hop), avg 3.1 agent steps,
  **strategy_change_rate 60%**, ~15k tokens/question, ~91 s/question.
- Misses: aggregation questions → honest refusals (need corpus-wide evidence);
  superlatives → near-misses ("Women's marathon" vs "Men's marathon") while the
  graph covers only ~6% of the corpus. Expect major gains after full extraction.
- Fixes during build: AgentState needed declared keys for final_answer/final_chunks
  (LangGraph drops undeclared ones); metrics moved to a runtime singleton
  (non-serializable objects don't survive state channels); citation post-mapping
  (block numbers → chunk_ids); supervisor prompt tolerant of non-retrieval trace
  entries; score_question accepts summarized metrics dicts.

### Quota reality (blocks full extraction)

Free-tier daily caps: gemini-3.5-flash-lite = 500/day/key (exhausted by the
full extraction run at 1,109/17,193 chunks); gemini-3.1-flash-lite switched in
(fresh 500/day/key, now partially used by agentic tests). Full extraction
needs ~16k more calls — options: paid tier, or ~1k chunks/day across model
pools (2+ weeks), or reduce corpus scope. EXTRACTION PAUSED pending decision.

## 11. Next Steps

1. **Resolve extraction quota** (user decision): paid tier / multi-day free
   pools / reduced corpus scope.
2. **Finish full-corpus extraction + graph load** (checkpointed, resumes free).
3. **Full eval**: all 100 public questions through all 3 pipelines →
   README comparison table (accuracy, latency, tokens, agent trace metrics).
4. Optional: real community detection; RAGResult adapters for Pipelines 1/2.

## 12. FINAL EVALUATION RESULTS (gpt-5-mini, all 3 pipelines — 2026-09-30)

All 6 runs saved under `results/<pipeline>/<set>_<timestamp>.json` (self-describing:
model, provider, config, summary, per-question records). Model: gpt-5-mini via
Azure (LLM_PROVIDER=azure), locked for every call in every pipeline.

### Public set (100 questions, locally scoreable)

| Pipeline | Accuracy | Avg total tokens/q | Avg context tokens | Steps/q | Strategy change |
|---|---|---|---|---|---|
| Vector RAG (P1) | **63%** | 3,328 | 2,301 | — | — |
| GraphRAG (P2) | 50% | 3,383 | — | — | — |
| Agentic GraphRAG (P3) | 58% | 12,794 | — | 1.95 | 28% |

### Hidden set (50 questions, blind — organizers hold gold answers)

Runtime metrics only (accuracy not scoreable locally):
- Vector: 3,348 tokens/q
- Graph: 3,521 tokens/q
- Agentic: 15,265 tokens/q, 2.46 steps/q, 44% strategy changes

### Findings
1. Vector baseline is strong on this corpus (Wikipedia factoid style): 63%.
2. Pure GraphRAG (50%) loses accuracy to honest refusals on aggregation
   questions whose gold evidence spans many docs, and to entity-matching
   misses at 180k-vertex scale.
3. Agentic (58%) recovers much of GraphRAG's loss via adaptive tool switching,
   but pays 3.8x the token cost vs vector — at this corpus scale the extra
   reasoning does NOT yet pay for itself in accuracy. The trade flips if the
   question mix favors multi-hop/relational questions (the demo scenario).
4. Answer style matters: strict grounding rules cause honest refusals that
   count as misses; an accuracy-vs-grounding tradeoff curve could be measured.

### Scale-up bugs found & fixed during eval bring-up
- Chroma single-get cap: --full only ever saw the first 5,000 chunks (now paginated).
- Entity matcher at 180k scale: bare-cosine threshold garbage-collected matches
  (canoeing question → Sailing chunks); fixed with sim>=0.90-or-overlap>=0.6 rule,
  best-fuzzy-only seeding, phrase-length guard, and canonical event-name prompt.
- Wrong-discipline evidence (C-2 for a K-2 question): identity-verification rules
  added to evaluator + answer prompts.
- Per-seed fan-out caps to prevent broad-seed dilution; partial-answer behavior.
