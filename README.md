# Advanced RAG with LangGraph

Self-correcting RAG pipeline built with LangGraph and contextual retrieval. 
Query rewriting, hybrid retrieval, Cohere reranking, and grounded self-check 
with retry loops.

**Read the [architecture document](docs/architecture.md)** for design rationale, 
the StateGraph diagram, measured eval results, and known weaknesses.

## Highlights

- 100% answer match and citation accuracy on a 30-question eval
- Anthropic's contextual retrieval implemented end-to-end (Haiku-generated 
  chunk prefixes, +recall on ambiguous queries)
- LangGraph StateGraph with conditional routing and bounded retry loop
- Side-by-side vanilla vs contextual Chroma indices for ablation
- Cost-conscious model assignment: 80% Haiku, 20% Sonnet

## Quick start

See [Reproducing the results](docs/architecture.md#reproducing-the-results) 
in the architecture doc.

## Tech stack

LangGraph · LangChain · Anthropic Claude · Cohere · Chroma · Python 3.12# Advanced_Rag_LangGraph
Self-correcting RAG pipeline built with LangGraph and contextual retrieval. Query rewriting, hybrid retrieval, Cohere reranking, and grounded self-check with retry loops.
