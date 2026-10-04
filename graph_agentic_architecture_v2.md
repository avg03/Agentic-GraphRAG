# GraphRAG + Agentic GraphRAG — Hierarchical Retrieval Architecture V2

## 1. Purpose

This document defines the revised retrieval architecture for the two pipelines:

- **Pipeline 2 — GraphRAG**: strengthen graph retrieval so it can answer both local relational questions and broader sibling/collection questions.
- **Pipeline 3 — Agentic GraphRAG**: build on the improved GraphRAG and Vector RAG tools so the agent can reason about query scope, select relevant graph roots/contexts, evaluate evidence, and loop when the first investigation is incomplete.

The current Vector RAG pipeline remains unchanged and acts as the dense/BM25/reranker baseline and as a fallback tool for Agentic GraphRAG.

The core problem being solved is:

> The current graph can retrieve a node's local neighborhood and its supporting chunks, but many corpus-level questions require discovering multiple sibling events that share a higher-level topic/context.

Example:

> "Give me the winners from the 2008 Olympics."

A local event-centric graph can easily answer:

```text
Event A -> RESULTED_IN -> Winner A
```

but may fail to discover:

```text
Olympics 2008
    -> Event A -> Winner A
    -> Event B -> Winner B
    -> Event C -> Winner C
    -> ...
```

The revision therefore introduces **hierarchy-aware context retrieval** while preserving the existing event-centric ontology and TigerGraph grounding design.

---

## 2. Current Architecture and Bottleneck

### Current GraphRAG flow

```text
1. Query entity extraction
   Question
      -> Gemini
      -> mentions/canonical names

2. Entity matching
   mention
      -> exact normalized-name match
      -> BGE fuzzy match
      -> seed vertices

3. Graph traversal
   seed vertex
      -> /edges/{type}/{id}
      -> direct MENTIONED_IN -> Chunk
      -> reverse event edge -> ActionEvent -> MENTIONED_IN -> Chunk

4. Chunk lookup
   chunk IDs
      -> ChromaDB
      -> original text + metadata

5. Ranking + gate
   support score
      -> BGE cross-encoder
      -> if insufficient: dense top-up

6. Answer
   Gemini -> grounded answer
```

### Main limitation

The current traversal is effectively **seed-local** and capped:

```text
seed -> local edges -> ActionEvent -> Chunk
```

It does not explicitly model or resolve a **higher-level scope** shared by sibling events.

For a question that asks for a set/collection, the retriever needs to answer a different question first:

> "Which group of events belongs to the requested scope?"

Only then should it traverse those events to their outcomes and evidence chunks.

### Important design conclusion

A concatenated metadata string such as:

```text
a-b-c
```

is useful as a human-readable lineage representation, but it should **not** be the only machine-readable representation of hierarchy.

Use structured lineage/context fields and a queryable context index. Otherwise the system still has to perform string interpretation before it can discover siblings.

---

## 3. Revised Design Principles

### 3.1 Preserve the existing ontology

Keep the approved TigerGraph core:

- ActionEvent
- Person
- Company
- Organization
- Team
- Country
- Location
- Venue
- Sport
- Outcome
- Entity
- Chunk

Keep the existing edge families:

- INITIATED_BY
- TOOK_PLACE_AT
- AFFECTED
- RESULTED_IN
- MENTIONED_IN

Do **not** replace the existing graph with a new ontology just to solve sibling retrieval.

### 3.2 Add hierarchy as retrieval metadata/context

Every important graph node should carry enough information to associate it with higher-level scopes.

Conceptually:

```text
root_context
context_path
context_keys
context_level
```

Example for an event:

```json
{
  "root_context": "Olympics 2008",
  "context_path": [
    "Olympics 2008",
    "Canoeing",
    "Canoe sprint",
    "Men's K-2 1000 metres"
  ],
  "context_keys": [
    "olympics:2008",
    "sport:canoeing",
    "discipline:canoe_sprint"
  ],
  "context_level": "event"
}
```

The exact storage format can be adapted to the existing ingestion code. The important property is that context is **structured and queryable**.

### 3.3 Separate discovery from traversal

Use two distinct operations:

```text
Context discovery
    -> Which graph roots / event groups are relevant?

Graph traversal
    -> What relationships and evidence exist inside those groups?
```

This prevents the TigerGraph traversal layer from trying to infer corpus-wide scope from one local seed.

### 3.4 Specific-first expansion

Do not blindly traverse every metadata component as an independent root.

For example, for:

```text
Olympics 2008 -> Canoeing -> Men's K-2 1000 metres
```

search in this order:

```text
1. exact event context
2. event-group / discipline context
3. edition-level context: Olympics 2008
4. broader context only if evidence is still insufficient
```

This prevents massive duplicate retrieval from broad roots.

---

# 4. Revised GraphRAG Architecture

## 4.1 New end-to-end flow

```text
USER QUERY
    |
    v
[1] Query Entity + Scope Extraction
    |
    |-- explicit entities
    |-- requested scope
    |-- question type
    |-- candidate context keys
    v
[2] Entity Matching
    |
    |-- exact entity matches
    |-- fuzzy entity matches
    v
[3] Context Resolution / Context Index
    |
    |-- match query/context keys to indexed graph nodes
    |-- identify candidate roots / event groups
    |-- rank by specificity
    v
[4] Hierarchy-Aware Graph Traversal
    |
    |-- local traversal for explicit entities
    |-- scope traversal for collection queries
    |-- expand candidate sibling events
    v
[5] Evidence Chunk Collection
    |
    |-- ActionEvent -> MENTIONED_IN -> Chunk
    |-- Outcome -> MENTIONED_IN -> Chunk
    |-- Person/Team/etc. -> MENTIONED_IN -> Chunk
    v
[6] Evidence Aggregation + Ranking
    |
    |-- graph support
    |-- context relevance
    |-- event diversity
    |-- query reranker
    v
[7] Sufficiency / Coverage Gate
    |
    |-- sufficient -> answer
    |-- insufficient -> targeted dense top-up
    v
[8] Gemini Grounded Answer
```

---

## 4.2 Stage 1 — Query Entity + Scope Extraction

The current Gemini extraction call should be extended so it returns both:

### Entity mentions

Example:

```json
[
  "Hungary",
  "Canoeing at the 2012 Summer Olympics – Men's K-2 1000 metres"
]
```

### Scope information

For:

> "total winner names from Olympics 2008"

return something conceptually like:

```json
{
  "question_type": "collection_aggregation",
  "entities": ["Olympics 2008"],
  "scope": {
    "root": "Olympics 2008",
    "breadth": "all_events",
    "target": "winner"
  },
  "candidate_context_keys": [
    "olympics:2008",
    "year:2008"
  ]
}
```

The model should identify the intended **scope**, not invent graph facts.

For a local query:

> "Who won the men's K-2 1000 metres in 2012?"

the scope could be:

```json
{
  "question_type": "local_fact",
  "breadth": "single_event",
  "target": "winner"
}
```

This distinction is important because the retrieval strategy should depend on query type.

---

# 5. Context Index

## 5.1 Why an additional context index is needed

The existing entity index is:

```text
mention -> graph vertex
```

Add a related index:

```text
context key / context phrase -> relevant graph vertex IDs
```

Example:

```text
"olympics:2008"
    -> ActionEvent:event_1
    -> ActionEvent:event_2
    -> ActionEvent:event_3
    -> ...
```

This gives the system a deterministic way to discover siblings.

### Recommended implementation

Create a `context_index.py` module alongside `entity_index.py`.

It can use the same general strategy:

1. exact normalized context key / phrase match
2. semantic/BGE matching for paraphrases
3. deterministic ranking
4. return top-K candidate graph nodes

The index can be precomputed locally and cached to disk, just like the existing 180,959-node entity index.

The actual graph relationships still live in TigerGraph; the context index is a **candidate discovery layer**.

---

## 5.2 Context representation

### Do not rely only on:

```text
a-b-c
```

Prefer a representation like:

```json
{
  "context_path": [
    "Olympics 2008",
    "Canoeing",
    "Canoe sprint",
    "Men's K-2 1000 metres"
  ]
}
```

plus normalized keys:

```text
olympics:2008
sport:canoeing
discipline:canoe_sprint
event:mens_k2_1000_metres
```

You may also retain a compact display/diagnostic form:

```text
Olympics 2008 > Canoeing > Canoe sprint > Men's K-2 1000 metres
```

This should be treated as **lineage/provenance**, not as the retrieval primitive itself.

---

# 6. Revised TigerGraph Traversal

## 6.1 Two traversal modes

The GraphRAG retriever should have two explicit modes.

### Mode A — Local relational traversal

Used for narrow questions.

```text
seed entity
   |
   +--> direct MENTIONED_IN --> Chunk
   |
   +<-- event edge -- ActionEvent -- MENTIONED_IN --> Chunk
```

This preserves current behavior.

### Mode B — Scope/sibling traversal

Used for collection/aggregation questions.

```text
context root / candidate event set
          |
          v
   ActionEvent candidates
          |
          +--> RESULTED_IN --> Outcome
          |
          +--> MENTIONED_IN --> Chunk
          |
          +--> related entity edges
```

The key difference is that Mode B begins from a **set of candidate sibling events**, rather than a single entity seed.

---

## 6.2 Example: 2008 Olympics

Suppose the context index resolves:

```text
olympics:2008
```

to:

```text
Event A
Event B
Event C
...
```

The retriever then asks TigerGraph for relationships from that candidate set and collects:

```text
Event A -> RESULTED_IN -> Outcome A
Event B -> RESULTED_IN -> Outcome B
Event C -> RESULTED_IN -> Outcome C
```

and grounding:

```text
Event A -> MENTIONED_IN -> Chunk A
Event B -> MENTIONED_IN -> Chunk B
Event C -> MENTIONED_IN -> Chunk C
```

The final context therefore contains evidence from multiple sibling branches.

---

## 6.3 Avoid traversing every metadata component independently

The initial idea:

```text
a root
b root
c root
-> traverse each
```

can create large amounts of duplicate evidence.

Instead, assign a specificity rank:

```text
Level 4: exact event
Level 3: discipline/category
Level 2: sport
Level 1: edition/root
```

For a narrow query, start at the most specific level.

For an aggregation query, start from the requested collection root.

Only broaden when the current scope fails the coverage/sufficiency gate.

Example:

```text
Exact event -> enough evidence -> STOP

Exact event -> insufficient -> expand discipline

Discipline -> insufficient -> expand 2008 Olympics

2008 Olympics -> sufficient -> STOP
```

---

# 7. Retrieval Result Structure

The existing structure:

```python
{
    "chunk_id": {
        "support": 3,
        "via": [...]
    }
}
```

should be expanded to preserve scope information.

Conceptually:

```python
{
    "chunk_id": {
        "support": 3,
        "via": [...],
        "event_ids": [...],
        "context_keys": ["olympics:2008", "sport:canoeing"],
        "scope_level": "edition",
        "retrieval_mode": "scope"
    }
}
```

This is important for both debugging and Agentic GraphRAG.

The agent should be able to tell the difference between:

```text
"I found this chunk because it directly mentioned the query entity"
```

and:

```text
"I found this chunk because its ActionEvent belongs to the 2008 Olympics scope"
```

---

# 8. Improved Ranking

The current ranking is mainly:

```text
support
   -> cross-encoder relevance
```

Retain the cross-encoder, but add scope-aware signals.

Recommended ranking concept:

```text
FinalScore =
    query_relevance
  + graph_support
  + context_match
  + evidence_diversity
  - duplication_penalty
```

Where:

### query_relevance

How semantically relevant the chunk is to the actual question.

### graph_support

How many independent graph routes support it.

### context_match

How closely its lineage matches the requested scope.

### evidence_diversity

Reward evidence from distinct ActionEvents/sibling branches.

### duplication_penalty

Prevent many chunks from the same event from dominating the top-K.

This last point is especially important for collection questions.

For example, 20 chunks from one event should not outrank 10 chunks covering 10 distinct events simply because support count is higher.

---

# 9. Replace the Simple "3 Chunks" Sufficiency Gate

The existing rule:

```text
if usable_chunks < 3:
    dense top-up
```

is too weak for collection questions.

Three chunks may be enough for:

> "Where was Event X held?"

but obviously not enough for:

> "List all winners from the 2008 Olympics."

Introduce **query-aware coverage checks**.

### Local question

Check:

```text
required relation found?
required entity found?
evidence grounded?
```

### Collection question

Check:

```text
requested scope identified?
multiple sibling branches discovered?
required target relation found across branches?
number of distinct target entities increasing?
new retrieval steps still adding useful evidence?
```

The system should not claim mathematical completeness unless the graph contains enough information to establish the full scope.

Instead return something such as:

```text
scope_coverage = partial / substantial / high
```

and expose uncertainty when appropriate.

---

# 10. Agentic GraphRAG V2

## 10.1 Goal

Agentic GraphRAG should not merely repeat GraphRAG several times.

It should decide **which scope/root to investigate next and whether further investigation is justified**.

The agent uses one Gemini model for:

- planning
- query/scope analysis
- evidence evaluation
- supervision
- final answer generation

Tools remain deterministic application functions.

---

## 10.2 Proposed architecture

```text
USER QUERY
    |
    v
[Planner]
    |
    |-- question type
    |-- answer slots
    |-- entity mentions
    |-- scope hypotheses
    v
[Context Resolver]
    |
    |-- candidate roots
    |-- candidate sibling groups
    v
[Supervisor]
    |
    +----> GRAPH_LOCAL
    |
    +----> GRAPH_SCOPE
    |
    +----> VECTOR_SEARCH
    |
    +----> REFORMULATE
    |
    +----> STOP
    v
[Retrieval Tool]
    |
    +--> TigerGraph traversal
    +--> Chroma/BM25/dense baseline when selected
    v
[Evidence Evaluator]
    |
    |-- support
    |-- scope coverage
    |-- missing answer slots
    |-- contradictions
    |-- marginal gain
    v
[Supervisor Loop]
    |
    +--> more graph scope expansion
    +--> alternate root
    +--> vector fallback
    +--> stop
    v
[Context Builder]
    |
    v
[Gemini Answer Generator]
```

---

# 11. Agent State

Use explicit structured state.

```python
AgentState = {
    "question": str,
    "question_type": str,
    "answer_slots": [...],
    "entity_mentions": [...],
    "matched_seeds": [...],
    "candidate_contexts": [...],
    "investigated_contexts": [...],
    "graph_evidence": [...],
    "vector_evidence": [...],
    "chunks": [...],
    "evidence_scores": [...],
    "coverage": {...},
    "contradictions": [...],
    "step_count": int,
    "action_history": [...],
    "next_action": str,
    "stop_reason": str
}
```

The `investigated_contexts` field is critical.

It prevents the agent from repeatedly calling the same root:

```text
olympics:2008 -> already investigated
```

and forces it to choose another useful action.

---

# 12. Agent Action Space

Keep the action space small and explicit.

```text
GRAPH_LOCAL
GRAPH_SCOPE
VECTOR_SEARCH
REFORMULATE
STOP
```

### GRAPH_LOCAL

Investigate direct entity/event relationships.

### GRAPH_SCOPE

Investigate siblings belonging to a context root.

### VECTOR_SEARCH

Use the existing Pipeline 1 retrieval stack when graph evidence is missing, too sparse, or the ontology is insufficient.

### REFORMULATE

Generate an alternate entity/context phrase when matching failed or the current scope is ambiguous.

### STOP

Use when evidence is sufficient or no useful next action remains.

This action space is deliberately small to make the trace measurable and debuggable.

---

# 13. ReAct Loop Policy

The loop should follow:

```text
THINK
  -> choose action
ACT
  -> execute retrieval tool
OBSERVE
  -> inspect evidence
EVALUATE
  -> calculate coverage / missing slots / conflicts
DECIDE
  -> continue or stop
```

The LLM should never directly execute TigerGraph queries.

Instead:

```text
Gemini
   -> structured tool/action request
Application
   -> TigerGraph / Chroma / reranker
Application
   -> structured observation
Gemini
```

This preserves determinism and makes the trace auditable.

---

# 14. Example Agentic Trace for the Sibling Problem

Question:

> "Give me the winners from the 2008 Olympics."

### Step 1 — Planner

```text
question_type = collection_aggregation
scope = olympics:2008
answer_target = winner
```

### Step 2 — Context Resolver

```text
candidate_contexts:
1. olympics:2008
2. year:2008
3. Olympic Games 2008
```

### Step 3 — Supervisor

```text
action = GRAPH_SCOPE
reason = collection question requires sibling event discovery
```

### Step 4 — Graph Scope Search

Resolve `olympics:2008` to a set of candidate ActionEvents and retrieve:

```text
Event A -> Outcome A -> Chunk A
Event B -> Outcome B -> Chunk B
Event C -> Outcome C -> Chunk C
...
```

### Step 5 — Evaluator

```text
required target = winner
unique winner evidence = increasing
scope coverage = substantial
remaining uncertainty = whether all event branches are represented
```

### Step 6 — Supervisor

If evidence continues to add new events:

```text
action = GRAPH_SCOPE
reason = additional sibling branches may still contribute new winners
```

If a second scope key is effectively equivalent:

```text
action = STOP
reason = no new evidence gain / duplicate scope
```

If graph coverage is too sparse:

```text
action = VECTOR_SEARCH
reason = graph scope incomplete; use independent textual evidence
```

### Step 7 — Final Answer

Generate only from retrieved evidence, with chunk citations.

---

# 15. Critical Edge Cases

## 15.1 Ambiguous root

Example:

> "winners in 2008"

The system must determine whether the corpus context implies Olympics or another competition.

Do not blindly map `2008` to every year-related node.

Use entity context + question semantics + candidate context ranking.

If ambiguity remains, gather evidence from the highest-confidence candidates and expose uncertainty rather than inventing scope.

## 15.2 Graph miss does not mean fact absent

A missing path may result from:

- extraction failure
- ontology drop
- missing metadata
- entity-match failure
- insufficient traversal scope

Therefore:

```text
Graph miss
   !=
Fact absent
```

The Agent should consider `REFORMULATE` or `VECTOR_SEARCH` before concluding that the information is unavailable.

## 15.3 Hub explosion

Broad contexts may map to thousands of events.

Use:

```text
max_context_roots
max_candidate_events_per_root
max_chunks_per_event
max_total_chunks
max_steps
```

and prefer incremental expansion.

## 15.4 Duplicate siblings

The same event may be discovered from:

- an explicit entity seed
- a sport root
- an edition root
- another related entity

Deduplicate by stable `vertex_type + vertex_id` and by `chunk_id`.

## 15.5 Near-duplicate event names

Continue using the existing entity-index rule:

```text
exact normalized match first
fuzzy only with high confidence
best candidate only
```

Do not allow context expansion to override entity disambiguation.

## 15.6 Conflicting outcomes

If multiple chunks disagree:

```text
mark contradiction
retrieve more evidence if useful
avoid averaging/confusing the facts
surface the conflict in the final response when unresolved
```

## 15.7 Numerical/table answers

For times, scores, rankings, distances and similar values:

- use graph traversal to identify the event/entity scope
- retrieve the supporting chunk/table
- use the exact textual evidence for the final value

Graph structure should find the right branch; the chunk remains the primary grounding source for exact tabular values.

---

# 16. Recommended Module Changes

Suggested structure:

```text
rag_project/
|
+-- graphrag/
|   +-- entity_index.py          # existing
|   +-- context_index.py         # NEW
|   +-- traverse.py              # extend with local + scope traversal
|   +-- scope.py                 # NEW: scope/context resolution helpers
|   +-- rank.py                  # add context/diversity scoring
|   +-- retrieve.py              # NEW/UPDATED orchestration layer
|   +-- schemas.py               # structured retrieval results
|
+-- agentic/
|   +-- state.py
|   +-- planner.py
|   +-- supervisor.py
|   +-- evaluator.py
|   +-- tools.py                 # graph/vector tools
|   +-- workflow.py              # LangGraph workflow
|
+-- shared/
    +-- llm_client.py
    +-- metrics.py
    +-- prompts.py
    +-- models.py
```

Do not rewrite the working Vector RAG engine.

Expose its retrieval functionality as:

```python
vector_search(query, filters=None, top_k=...)
```

and expose the revised GraphRAG as:

```python
graph_search(
    query,
    seeds=None,
    context_keys=None,
    mode="auto",
    top_k=...
)
```

---

# 17. Suggested Graph Search API Semantics

The graph retriever should distinguish three inputs:

### seeds

Explicit entity/event vertices.

### context_keys

Higher-level scopes such as:

```text
olympics:2008
sport:canoeing
```

### mode

```text
local
scope
auto
```

`auto` selects based on the query type produced by the planner/query analyzer.

Example:

```python
graph_search(
    query="winners from the 2008 Olympics",
    context_keys=["olympics:2008"],
    mode="scope",
    top_k=30,
)
```

For a local question:

```python
graph_search(
    query="Who won the men's K-2 1000m event?",
    seeds=[...],
    mode="local",
    top_k=10,
)
```

---

# 18. Metrics to Add for the New Retrieval Strategy

Keep the existing hackathon metrics and add scope-specific metrics.

### Existing

- accuracy
- completeness
- groundedness
- context tokens
- input tokens
- output tokens
- total tokens
- latency
- retrieved chunks
- citations

### New GraphRAG metrics

```text
unique_seed_vertices
unique_context_roots
unique_action_events
unique_target_outcomes
unique_chunks
scope_coverage
retrieval_mode
sibling_expansion_count
duplicate_chunk_count
```

### New Agentic metrics

```text
step_count
actions_taken
contexts_investigated
strategy_changes
vector_fallback_count
reformulation_count
new_evidence_per_step
marginal_gain_per_step
stop_reason
```

This directly supports the hackathon objective of determining whether additional reasoning/retrieval steps are worth their cost.

---

# 19. Retrieval Stopping Criteria

The agent should stop for one of four reasons:

### A. Sufficient evidence

All required answer slots have supporting evidence and additional retrieval is not materially improving coverage.

### B. Diminishing returns

Recent retrieval actions produce no meaningful new entities/events/chunks.

### C. Hard budget

Maximum steps, chunks, latency, or token budget reached.

### D. No useful path remains

Candidate contexts are exhausted and vector fallback/reformulation also fails.

Always record the exact `stop_reason`.

---

# 20. Minimal Architectural Change vs Stronger Long-Term Change

## Minimal change — recommended first

Keep the existing TigerGraph schema.

Add:

```text
structured context metadata
context_index.py
scope-aware traversal
scope-aware ranking
query-aware sufficiency
Agentic GRAPH_SCOPE action
```

This is the lowest-risk change and directly addresses the sibling-retrieval failure.

## Stronger long-term option

If future experiments show that contextual relationships are central to the corpus, introduce explicit first-class context/edition nodes and graph edges such as:

```text
Olympics2008 -> HAS_EVENT -> ActionEvent
```

This turns the scope hierarchy into an actual graph structure.

However, this should be treated as a later ontology extension, not a prerequisite for the current fix.

---

# 21. Acceptance Tests

The revised GraphRAG should be tested with at least these query classes.

### Class A — Single-hop/local

```text
Who won Event X?
Where did Event X take place?
```

Expected: local traversal remains strong.

### Class B — Multi-hop relational

```text
Which person initiated Event X and where did it occur?
```

Expected: multiple relation families can be traversed and grounded.

### Class C — Sibling/collection

```text
Who were the winners across the 2008 Olympics events?
```

Expected: multiple sibling ActionEvents contribute evidence.

### Class D — Filtered collection

```text
Which 2008 Olympic canoeing events were won by Hungary?
```

Expected: context scope + entity filtering + event/outcome traversal.

### Class E — Exact table fact

```text
What was the winning time in Event X?
```

Expected: graph identifies the event; supporting chunk/table provides exact value.

### Class F — Graph miss

Use a fact that is present in the text corpus but has no extracted graph path.

Expected: GraphRAG can top-up or Agentic GraphRAG can switch to vector retrieval rather than declaring the fact absent immediately.

### Class G — Ambiguous entity

Use a name shared by multiple entities/events.

Expected: deterministic disambiguation and no blind traversal of the wrong branch.

---

# 22. Final Target Architecture

```text
                         USER QUERY
                             |
                             v
                +---------------------------+
                | Query / Scope Analyzer    |
                | Gemini                    |
                +---------------------------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Entity Mentions               Scope / Context
              |                             |
              v                             v
       Entity Index                  Context Index
              |                             |
              +--------------+--------------+
                             |
                             v
                 Candidate Graph Roots
                             |
                             v
                +-------------------------+
                | TigerGraph Traversal    |
                |                         |
                | Local mode              |
                | Scope/sibling mode      |
                +-------------------------+
                             |
                             v
                      Evidence Chunks
                             |
                             v
                 Context + Query Reranker
                             |
                             v
                    Coverage / Gate
                       /           \
                      /             \
                enough            missing
                   |                 |
                   v                 v
                Answer          Vector Search
                   |                 |
                   +-------+---------+
                           |
                           v
                    Grounded Answer
```

For the Agentic version:

```text
                         USER QUERY
                             |
                             v
                          PLANNER
                             |
                             v
                       SUPERVISOR
                    /      |       \
                   /       |        \
          GRAPH_LOCAL  GRAPH_SCOPE  VECTOR
                   \       |        /
                    \      |       /
                       EVALUATOR
                           |
                 enough? /       \ missing?
                        /         \
                      STOP      next action
                                  |
                                  v
                              SUPERVISOR
                                  |
                              (loop with
                             hard limits)
                                  |
                                  v
                              ANSWER
```

---

# 23. Core Insight

The architectural improvement is not simply:

```text
a -> b -> c
```

The important change is:

```text
QUERY
  -> identify scope
  -> resolve scope to candidate sibling roots/events
  -> traverse each relevant branch
  -> aggregate evidence across branches
  -> evaluate coverage
  -> expand only when useful
```

In other words:

> **Use metadata to discover graph scope; use TigerGraph to traverse relationships inside that scope; use the agent to decide which scope to investigate next.**

This keeps the graph authoritative for relationships, keeps chunks authoritative for textual evidence, keeps Vector RAG as a useful fallback, and gives Agentic GraphRAG a principled mechanism for solving questions that require sibling/collection retrieval.
