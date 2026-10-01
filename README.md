# GraphRAG Hackathon — Three Progressive Retrieval Pipelines

Three retrieval pipelines over the same Wikipedia corpus, compared experimentally:

1. **Vector RAG** (`retrieval/`) — BM25 + dense retrieval (BGE) + cross-encoder rerank + LLM answer
2. **GraphRAG** (`graphrag/`) — TigerGraph knowledge graph (typed event-centric schema) with entity/relation extraction, graph traversal, and supporting-chunk grounding
3. **Agentic GraphRAG** (`agentic/`) — LangGraph investigation layer that plans, decomposes questions into answer slots, routes between graph/vector/community tools, evaluates evidence per slot, and stops when sufficient

All three share one LLM funnel (`shared/llm_client.py` — provider-switchable: Gemini / Azure / Groq / OpenRouter / NVIDIA NIM), one prompt library, one metrics contract (`shared/result.py`), and one run-persistence store (`shared/run_store.py`).

## Results snapshot (gpt-5-mini, 100 public questions)

| Pipeline | Accuracy | Avg total tokens/q |
|---|---|---|
| Vector RAG | **63%** | 3,328 |
| GraphRAG | 50% | 3,383 |
| Agentic GraphRAG | 58% (1.95 steps, 28% strategy changes) | 12,794 |

Full per-question records: `results/<pipeline>/<eval_set>_<timestamp>.json`.

## Repo layout

```
shared/      LLM client, prompts, metrics, result contract, run store
ingestion/   corpus loading, chunking, embedding, one-time Chroma ingestion
retrieval/   Pipeline 1: BM25, dense, reranker, full vector pipeline + eval
database/    ChromaDB client
graphrag/    Pipeline 2: ontology, extraction (checkpointed), dedup, TigerGraph load, retrieval pipeline + eval
agentic/     Pipeline 3: planner, tools, evaluator, supervisor, LangGraph loop + eval
Data/        corpus + eval sets (corpus.jsonl not committed — see below)
results/     saved evaluation runs (self-describing JSON documents)
context.md   project spec / architecture source of truth
progress.md  full build log: what was built, errors, fixes, findings
```

## Setup

```bash
python -m venv ragvenv && ragvenv/Scripts/activate   # Windows
pip install -r requirements.txt
copy .env.example .env   # then fill in keys (see below)
```

`.env` keys: `GOOGLE_API_KEY` (Gemini), `AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT`,
`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`,
`TIGERGRAPH_HOST/GRAPHNAME/SECRET`, and `LLM_PROVIDER` (`auto` | `gemini` | `azure` | `groq` | `openrouter`).
The corpus (`Data/corpus.jsonl`) is not committed — place it in `Data/` before ingesting.

## Running

```bash
# 0. One-time corpus ingestion -> ChromaDB
python -m ingestion.run_ingest --full

# 1. GraphRAG ingestion: chunks -> Gemini entity/relation extraction -> TigerGraph
python -m graphrag.ingest_pipeline --full --eval-first --skip-load   # extraction only
python -m graphrag.schema && python -m graphrag.ingest_pipeline --full --eval-first  # + graph load

# 2. Ask a question (any pipeline)
python -m retrieval.vector_pipeline "question"
python -m graphrag.retrieval_pipeline "question"
python -m agentic.agent "question" --trace

# 3. Evaluations (saved to results/)
python eval_all.py   # all 3 pipelines x public/hidden, sequential, saved
```

See `progress.md` for the complete build log and `context.md` for the original spec.
