# Agentic GraphRAG Architecture

## 1. Project Context

This system is the **third and primary pipeline** of the hackathon submission.

The project contains three pipelines over the same corpus:

1. **Vector RAG** — BM25 + dense retrieval + reranking + LLM generation.
2. **GraphRAG** — TigerGraph-based knowledge graph retrieval using entities, relationships, ActionEvent-centric graph structure, and supporting chunks.
3. **Agentic GraphRAG** — an agentic investigation layer built **on top of the existing Vector RAG and GraphRAG engines**.

The first two engines are already built and should remain intact. Agentic GraphRAG should **orchestrate/reuse** them rather than reimplement them.

The main objective of Agentic GraphRAG is not simply to produce a better answer. It should demonstrate that the system can:

- understand a complex question,
- decompose it when necessary,
- choose an appropriate retrieval method,
- inspect intermediate evidence,
- detect missing or weak evidence,
- perform additional retrieval/reasoning when required,
- combine graph and textual evidence,
- stop when sufficient evidence has been collected,
- expose a trace explaining what the system did.

---

# 2. Core Design Principle

> **GraphRAG finds connected evidence. Agentic GraphRAG decides what evidence is still missing and what to do next.**

The agent is therefore an **orchestrator**, not another database or another retriever.

The architecture should use **one LLM only: Gemini**.

Gemini performs the following logical roles:

- query planner,
- query/subquery decomposition,
- tool routing,
- intermediate evidence evaluation,
- supervision/next-action selection,
- final answer generation.

These are logical workflow nodes, not separate LLM models.

---

# 3. Existing Engines Being Reused

## 3.1 Vector RAG Engine

Already implemented.

Current retrieval stack:

```text
User Query
   ↓
BM25 + Dense Retrieval
   ↓
Reranker
   ↓
Top-ranked Chunks
   ↓
LLM
   ↓
Answer
```

The Agentic GraphRAG system should expose the existing Vector RAG retrieval capability as a reusable tool such as:

```python
vector_search(query, top_k=...)
```

The agent should not duplicate BM25, dense retrieval, or reranking logic.

---

## 3.2 GraphRAG Engine

Already implemented/almost complete.

Graph database: **TigerGraph**.

The graph contains:

- typed entity vertices,
- `ActionEvent` vertices,
- `Chunk` vertices,
- event-centric relationship edges,
- `MENTIONED_IN` grounding edges.

The GraphRAG engine should expose a reusable graph retrieval tool such as:

```python
graph_search(query_or_subquery, ...)
```

The tool should return structured graph evidence and supporting chunk references.

---

# 4. Agentic GraphRAG High-Level Architecture

```text
                         USER QUERY
                              │
                              ▼
                    ┌───────────────────┐
                    │   GEMINI PLANNER  │
                    │ Query Understanding│
                    │ + Decomposition   │
                    └─────────┬─────────┘
                              │
                       Answer Slots /
                       Investigation Tasks
                              │
                              ▼
                    ┌───────────────────┐
                    │  RETRIEVAL ROUTER │
                    │   (Gemini decides)│
                    └─────────┬─────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
          Graph Search   Vector Search   (Optional)
          TigerGraph      Existing RAG   Community Search
                 │            │            │
                 └────────────┼────────────┘
                              ▼
                    ┌───────────────────┐
                    │ EVIDENCE EVALUATOR│
                    │      (Gemini)     │
                    └─────────┬─────────┘
                              │
                    Is evidence sufficient?
                        /               \
                      YES               NO
                       │                 │
                       ▼                 ▼
                CONTEXT BUILDER   ┌───────────────────┐
                       │           │    SUPERVISOR     │
                       │           │      (Gemini)     │
                       │           └─────────┬─────────┘
                       │                     │
                       │             Next action / retry /
                       │             reformulate / switch tool
                       │                     │
                       │                     └──────────► RETRIEVE
                       │
                       ▼
                ANSWER GENERATOR
                     (Gemini)
                       │
                       ▼
                      END
```

---

# 5. Logical Workflow Nodes

The system should be implemented as a LangChain/LangGraph-style workflow with the following logical nodes.

## Node 1 — Planner

### Responsibility

Understand the user question and decide whether it requires decomposition.

For complex questions, produce **answer slots / investigation tasks** rather than arbitrary subqueries.

### Example

Question:

> Which athletes from Hungary won medals in the 2012 men's K-2 1000m event, where was it held, and what were their times?

Planner output:

```json
{
  "complexity": "high",
  "answer_slots": [
    {
      "id": "s1",
      "question": "Which Hungarian athletes/teams won medals?",
      "type": "relational"
    },
    {
      "id": "s2",
      "question": "Where was the event held?",
      "type": "relational"
    },
    {
      "id": "s3",
      "question": "What were the relevant winning/medal times?",
      "type": "exact_fact"
    }
  ]
}
```

### Design rule

Do **not** force every question into 2–3 subqueries. Decompose only when the question contains multiple independently answerable requirements.

---

# 6. Retrieval Router

Gemini chooses the most appropriate retrieval method for each answer slot.

Possible actions:

```text
GRAPH_SEARCH
VECTOR_SEARCH
COMMUNITY_SEARCH (if implemented)
STOP / DIRECT_ANSWER
```

Examples:

| Question Type | Preferred Tool |
|---|---|
| Relationship / multi-hop | Graph Search |
| Exact textual detail / descriptive passage | Vector Search |
| Broad corpus-level overview | Community Search |
| Mixed question | Graph + Vector |

The router should not blindly invoke all tools.

---

# 7. Tool: Graph Search

## Purpose

Use the existing TigerGraph GraphRAG engine to investigate relationships.

### Flow

```text
Subquery
   ↓
Entity identification
   ↓
TigerGraph GSQL traversal
   ↓
Paths / entities / relationships
   ↓
Supporting Chunk IDs
   ↓
Graph Evidence
```

### Returns

Graph search should return structured data such as:

```json
{
  "tool": "graph_search",
  "entities": [],
  "paths": [],
  "relationships": [],
  "chunk_ids": [],
  "status": "success"
}
```

---

# 8. Tool: Vector Search

Reuse the existing Vector RAG retrieval stack.

### Flow

```text
Subquery
   ↓
BM25 + Dense Retrieval
   ↓
Reranker
   ↓
Supporting chunks
```

The tool should return the same structured retrieval information already produced by the Vector RAG engine.

Example:

```json
{
  "tool": "vector_search",
  "chunks": [],
  "scores": [],
  "citations": [],
  "status": "success"
}
```

---

# 9. Evidence Evaluator

This is a critical component.

Do not reduce evaluation to one arbitrary confidence score.

The evaluator should inspect whether each **answer slot** is actually supported.

## Example Evidence Matrix

```text
                         S1       S2       S3
Winner/Athlete           ✅       —        —
Venue                    —        ✅       —
Exact Time               —        —        ❌
```

### Evaluator output

```json
{
  "overall_status": "partial",
  "slots": {
    "s1": {
      "status": "supported",
      "support_score": 0.91
    },
    "s2": {
      "status": "supported",
      "support_score": 0.88
    },
    "s3": {
      "status": "missing",
      "support_score": 0.31
    }
  },
  "contradiction": false,
  "missing": ["exact medal time"],
  "recommended_action": "VECTOR_SEARCH"
}
```

The evaluator should consider:

- relevance,
- relation correctness,
- textual support,
- completeness,
- contradictions,
- whether the evidence answers the specific slot.

---

# 10. Supervisor

The Supervisor is another **logical role performed by Gemini**.

Its job is to decide what happens after evidence evaluation.

## Allowed actions

```text
CONTINUE
GRAPH_SEARCH
VECTOR_SEARCH
COMMUNITY_SEARCH
REFORMULATE
STOP
```

Example:

```json
{
  "action": "VECTOR_SEARCH",
  "target": "exact winning time",
  "reason_code": "missing_required_fact"
}
```

### Important

The supervisor should not continue simply because a score is below a threshold.

It should consider:

- missing answer slots,
- weak evidence,
- incorrect relationship type,
- entity ambiguity,
- contradictions,
- graph miss vs actual absence,
- duplicate retrieval,
- maximum step budget.

---

# 11. Context Builder

Once sufficient evidence has been collected, convert the structured evidence into a compact grounded context for the final Gemini call.

The context should contain:

- relevant graph paths,
- relevant relationships,
- supporting chunks,
- source metadata,
- chunk IDs / citations.

Duplicate chunks should be removed.

Example structure:

```text
GRAPH EVIDENCE
1. Event → RESULTED_IN → Outcome
2. Event → TOOK_PLACE_AT → Venue

SUPPORTING SOURCES
[Q303623_004]
<chunk text>

[Q303623_006]
<chunk text>
```

---

# 12. Answer Generator

Final Gemini call.

Responsibilities:

- answer only from collected evidence,
- preserve citations,
- distinguish supported facts from unavailable information,
- avoid hallucinating missing facts.

If the investigation could not establish a fact, the answer should explicitly say that the corpus did not provide sufficient evidence.

---

# 13. Agent State

The shared state should contain enough information for the agent to investigate and supervise itself.

Recommended state:

```python
AgentState = {
    "question": str,

    "answer_slots": list,
    "completed_slots": list,
    "unresolved_slots": list,

    "plan": dict,
    "next_action": str,

    "graph_evidence": list,
    "vector_evidence": list,
    "community_evidence": list,

    "chunks": list,
    "citations": list,

    "evidence_scores": dict,
    "contradictions": list,

    "investigation_history": list,

    "step_count": int,
    "max_steps": int,

    "stop_reason": str,
    "answer": str
}
```

---

# 14. Control Loop

The core investigation loop is:

```text
PLAN
 ↓
RETRIEVE
 ↓
EVALUATE
 ↓
SUPERVISE
 ↓
 ┌──────────────────────────────┐
 │                              │
 │ enough evidence?             │
 │                              │
 └───────┬──────────────┬───────┘
         │ YES          │ NO
         ▼              ▼
      ANSWER       choose next action
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
           GRAPH      VECTOR    REFORMULATE
             │          │          │
             └──────────┴──────────┘
                        │
                        └──────────► EVALUATE
```

---

# 15. Edge-Case-First Design

The main differentiator should be robust behavior on difficult questions.

## 15.1 Entity Not Found

```text
Graph Search
    ↓
No entity found
    ↓
Reformulate / Vector Search
    ↓
Still unsupported?
    ↓
STOP: insufficient evidence
```

Never hallucinate an entity that does not exist in the corpus.

---

## 15.2 Ambiguous Entity

Example:

```text
Paris
```

may represent multiple entities.

The system should inspect query context and evidence before choosing a graph node.

Do not blindly traverse the first matching name.

---

## 15.3 Graph Path Exists but Semantics Are Wrong

A path is not automatically evidence.

Example:

```text
Event → AFFECTED → Athlete
```

does not prove:

```text
Athlete won the event
```

The evaluator must validate **relationship semantics**, not only graph connectivity.

---

## 15.4 Missing Graph Edge

A graph miss does not necessarily mean the fact is absent.

Possible reasons:

- extraction failure,
- ontology limitation,
- dropped relation,
- entity resolution failure.

The agent should be able to fall back to Vector Search.

---

## 15.5 Contradictory Evidence

If different sources provide conflicting answers:

```text
Chunk A → Fact X
Chunk B → Fact Y
```

mark the state as:

```text
CONFLICT
```

Do not silently select one without evaluating the evidence.

---

## 15.6 Graph Explosion

Do not perform unrestricted traversal.

Use limits such as:

```python
max_hops = 2
max_paths = 20
max_entities = ...
```

Prefer relationship-filtered traversal.

---

## 15.7 Exact Numeric / Table Questions

The graph may identify the relevant event/person, while the exact value exists only in a results-table chunk.

Example:

```text
Graph
  ↓
Identify correct event / entity
  ↓
MENTIONED_IN
  ↓
Results chunk
  ↓
Vector/Text evidence
  ↓
Exact value
```

This is an important demonstration of complementary retrieval.

---

## 15.8 Partial Answer

If:

```text
S1 ✅
S2 ✅
S3 ❌
```

continue investigating S3 rather than generating the final answer immediately.

---

## 15.9 Duplicate Retrieval

If the same chunk is returned multiple times:


```text
Graph → Chunk A
Vector → Chunk A
Graph → Chunk A
```

Deduplicate before final context construction.

Also record duplicate retrieval in the trace for efficiency analysis.

---

## 15.10 Non-Converging Investigation

A hard stop is mandatory.

Suggested initial limits:

```python
MAX_STEPS = 5
MAX_GRAPH_CALLS = 3
MAX_VECTOR_CALLS = 2
```

These should be configurable.

---

# 16. Observability and Agent Trace

Every agent execution must record a structured trace.

Example:

```json
{
  "step": 1,
  "node": "planner",
  "action": "decompose",
  "reason": "multiple answer slots detected"
}
```

```json
{
  "step": 2,
  "node": "retrieval",
  "tool": "graph_search",
  "target": "event winner",
  "latency_ms": 143
}
```

```json
{
  "step": 3,
  "node": "evaluator",
  "status": "partial",
  "missing": ["exact winning time"]
}
```

```json
{
  "step": 4,
  "node": "supervisor",
  "action": "vector_search",
  "reason_code": "missing_exact_fact"
}
```

```json
{
  "step": 5,
  "node": "supervisor",
  "action": "stop",
  "reason_code": "all_required_slots_supported"
}
```

The trace should show **what the system did**, not hidden chain-of-thought.

---

# 17. Metrics

The same metrics must be produced for all three pipelines so they can be compared fairly.

## Accuracy / Quality

Measure:

- correctness,
- completeness,
- grounding in available evidence.

Use a fixed evaluation set with reference answers/expected evidence.

## Token Metrics

Record:

- context tokens,
- LLM input tokens,
- LLM output tokens,
- total tokens.

For Agentic GraphRAG additionally record:

- tokens per operation,
- tokens per step,
- total agent tokens.

## Latency

Record:

- total latency,
- graph retrieval latency,
- vector retrieval latency,
- planner latency,
- evaluator latency,
- supervisor latency,
- answer generation latency,
- time per tool call.

## Retrieval / Trace Metrics

Record:

- number of retrieval steps,
- retrieval methods selected,
- tools called,
- number of chunks retrieved,
- number of citations,
- strategy changes,
- number of loops,
- stopping reason.

---

# 18. Standardized Result Object

Every engine should return a comparable result object.

Example:

```python
RAGResult(
    pipeline="agentic_graphrag",
    question=query,
    answer=answer,
    citations=citations,
    retrieved_chunks=retrieved_chunks,
    context_tokens=context_tokens,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    total_tokens=total_tokens,
    latency_ms=latency_ms,
    metadata={
        "steps": steps,
        "tools_used": tools_used,
        "strategy_changed": strategy_changed,
        "stop_reason": stop_reason,
        "trace": trace,
    }
)
```

---

# 19. LangChain / LangGraph Implementation Shape

The implementation should map the conceptual nodes to workflow nodes.

```text
START
  ↓
planner
  ↓
router
  ↓
retrieval_tool
  ↓
evidence_evaluator
  ↓
supervisor
  ├───────────────→ answer_generator → END
  │
  ├───────────────→ graph_search ────┐
  │                                  │
  ├───────────────→ vector_search ───┤
  │                                  │
  └───────────────→ reformulate ─────┘
```

The supervisor creates the conditional loop.

---

# 20. Important Engineering Constraints

- **One LLM only:** Gemini.
- Reuse the existing Vector RAG engine.
- Reuse the existing GraphRAG engine.
- TigerGraph remains the graph database.
- Do not duplicate BM25/dense/reranker logic inside the agent.
- Do not build a multi-agent architecture unless a later experiment proves it necessary.
- Keep the action space small and explicit.
- Every investigation must have a maximum-step budget.
- Every final answer must be grounded in retrieved evidence.
- Preserve source/chunk IDs throughout the workflow.
- Deduplicate evidence before sending final context to Gemini.

---

# 21. Definition of Done

Agentic GraphRAG is considered complete when it can:

- [ ] receive a user question,
- [ ] classify/decompose complex questions,
- [ ] create answer slots,
- [ ] choose graph vs vector retrieval,
- [ ] query the existing TigerGraph GraphRAG engine,
- [ ] query the existing Vector RAG engine,
- [ ] evaluate intermediate evidence,
- [ ] detect missing evidence,
- [ ] switch retrieval strategy when necessary,
- [ ] detect contradictions/ambiguity,
- [ ] avoid unrestricted graph expansion,
- [ ] stop after sufficient evidence or a hard limit,
- [ ] construct grounded final context,
- [ ] generate a cited final answer,
- [ ] emit a complete agent trace,
- [ ] emit token/latency/retrieval metrics.

---

# 22. Main Demo Scenario

A strong demo should deliberately use a complex, multi-part question that exposes the limitations of a single retrieval method.

Example pattern:

> Identify the relevant athletes/teams, determine their relationship to the event, find the venue, and retrieve the exact result/time.

Expected agent behavior:

```text
Planner
  ↓
Decompose into answer slots
  ↓
Graph Search
  ↓
Identify connected entities/relationships
  ↓
Evidence Evaluation
  ↓
Exact numeric evidence missing
  ↓
Supervisor switches to Vector Search
  ↓
Results-table chunk retrieved
  ↓
Evidence Evaluation
  ↓
All slots supported
  ↓
STOP
  ↓
Grounded Answer
```

This demonstrates the central value of the agentic design: **the system changes strategy based on what it learns during investigation.**

---

# 23. Guiding Principle for Future Implementation

When implementing a new feature, ask:

> **Does this make the agent better at deciding what to investigate next, or does it merely add complexity?**

Prefer the smallest architecture that demonstrates:

```text
Planning
+ Tool Selection
+ Evidence Evaluation
+ Adaptive Retrieval
+ Reliable Stopping
+ Grounded Answering
+ Observable Metrics
```

The primary goal is a robust and measurable **Agentic GraphRAG**, not a large number of agents or complicated workflow nodes.
