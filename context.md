# GraphRAG Hackathon — Project Memory & Context

> **Purpose:** This README acts as the persistent context/memory for the coding IDE. Treat it as the source of truth for architecture, objectives, and implementation decisions.

---

## Project Objective

Build **three progressively advanced retrieval pipelines** over the **same corpus** and compare them experimentally.

### Pipeline 1 — Vector RAG

* BM25 + Dense Retrieval + Reranker
* Retrieve semantically relevant chunks
* Generate grounded answers

### Pipeline 2 — GraphRAG

* Build a knowledge graph using **TigerGraph**
* Extract entities and relationships from chunks
* Retrieve context through graph traversal
* Return supporting chunks linked from the graph

### Pipeline 3 — Agentic GraphRAG (**Primary Pipeline**)

This is the main deliverable.

The agent should:

1. Understand the user query.
2. Plan the investigation.
3. Choose the appropriate retrieval method.
4. Evaluate intermediate evidence.
5. Perform additional retrieval if necessary.
6. Stop once sufficient evidence is collected.
7. Generate a grounded answer.

The first two pipelines are baselines; **Agentic GraphRAG is the flagship system.**

---

# Tech Stack

## LLM (Use only one model)

**Gemini** is the only LLM used throughout the project.

Gemini is responsible for:

* Entity extraction
* Relationship extraction
* Query planning
* Tool routing
* Evidence evaluation
* Final answer generation

No separate planner model or supervisor model should be introduced.

---

## Databases

### Vector RAG

* ChromaDB
* BM25
* Reranker

### GraphRAG

* TigerGraph (mandatory graph database)

---

# Core Architecture

## Pipeline 1

User Query

↓

BM25 + Dense Retrieval

↓

Reranker

↓

Top Chunks

↓

Gemini

↓

Answer

---

## Pipeline 2

Corpus Chunks

↓

Gemini Entity Extraction

↓

Triples (Entity–Relation–Entity)

↓

TigerGraph

↓

Graph Traversal

↓

Supporting Chunk IDs

↓

Chunk Lookup

↓

Gemini

↓

Answer

---

## Pipeline 3

User Query

↓

Gemini Planner

↓

Choose Retrieval Tool

↓

Graph / Vector / Community Retrieval

↓

Evaluate Evidence

↓

More Retrieval? (if needed)

↓

Context Fusion

↓

Gemini

↓

Answer

---

# Metrics (Mandatory)

Every pipeline must output the same evaluation metrics.

## Performance Metrics

* Total latency (ms)
* Retrieval latency
* LLM generation latency
* Total execution time

## Token Metrics

* Context tokens
* Input tokens
* Output tokens
* Total tokens

These should come from Gemini usage metadata whenever possible.

## Retrieval Metrics

* Number of retrieved chunks
* Number of citations
* Top-K retrieved
* Retrieval scores (where applicable)

## Agentic Metrics (Pipeline 3 only)

* Number of reasoning steps
* Retrieval methods selected
* Tools invoked
* Time per tool
* Tokens consumed per tool
* Total tools used
* Whether strategy changed during execution
* Final stopping reason

The objective is to evaluate whether the additional reasoning is worth the extra latency and token cost.

---

# Shared Components

These modules must be reused across all pipelines.

```text
shared/
│
├── llm_client.py      # Gemini wrapper
├── metrics.py         # Token + latency collection
├── prompts.py         # Prompt templates
└── models.py          # Shared dataclasses
```

Only retrieval logic should differ between pipelines.

---

# Graph Schema (TigerGraph)

## Vertices

* Entity
* Chunk

## Edges

* RELATION (entity → entity)
* MENTIONED_IN (entity → chunk)

Every entity must connect to the chunk from which it was extracted. This preserves grounding and allows supporting evidence retrieval.

---

# Project Philosophy

This is **not** three separate projects.

It is one retrieval system with progressively stronger retrieval capabilities:

Vector RAG → GraphRAG → Agentic GraphRAG

Each pipeline should answer the same benchmark questions so their accuracy, latency, and token efficiency can be compared fairly.

---

# Current Status

* [x] Vector RAG retrieval completed
* [x] BM25 + Dense + Reranker implemented
* [x] Environment and Gemini API configured
* [ ] GraphRAG (TigerGraph)
* [ ] Agentic GraphRAG
* [ ] Unified metrics integration
* [ ] Final evaluation & README comparison

---

# Important Constraints

1. Use **Gemini only** for all LLM tasks.
2. Use **TigerGraph** as the graph database.
3. Keep the LLM client reusable across all engines.
4. Collect identical metrics for every pipeline.
5. Agentic GraphRAG is the primary submission and should demonstrate planning, routing, evidence evaluation, and grounded reasoning.
