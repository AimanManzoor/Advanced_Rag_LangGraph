"""StateGraph wiring — connects the six nodes into a runnable RAG pipeline.

The graph supports:
  - Linear progression: rewrite_query → route → retrieve → rerank → generate → self_check
  - Conditional routing: route node splits to retrieve OR tool_use based on query type
  - Retry loops: low-confidence answers loop back to rewrite_query (capped at 2 attempts)

Run: python -m src.graph "your question here"
"""
from langgraph.graph import StateGraph, START, END

from src.state import RAGState
from src.nodes import (
    rewrite_query,
    route as route_node,
    retrieve,
    rerank,
    generate,
    self_check,
    tool_use,
)


# ============================================================
# Conditional edge functions
# These return a string key that maps to a destination node.
# ============================================================

def route_branch(state: RAGState) -> str:
    """After route node — pick simple retrieval or multi-hop tool use."""
    return state.get("route", "simple")


def confidence_branch(state: RAGState) -> str:
    """After self_check — decide whether to retry, or exit the loop."""
    confidence = state.get("confidence", 0.0)
    attempts = state.get("attempts", 0)

    # Hard cap: never loop more than 2 times. Production safety.
    if attempts >= 2:
        return "done"

    # If the answer wasn't well-grounded and we have attempts left, retry.
    if confidence < 0.7:
        return "retry"

    return "done"


# ============================================================
# Build the graph
# ============================================================

def build_graph():
    """Construct and compile the RAG StateGraph."""
    g = StateGraph(RAGState)

    # Register every node
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve)
    g.add_node("tool_use", tool_use)
    g.add_node("rerank", rerank)
    g.add_node("generate", generate)
    g.add_node("self_check", self_check)

    # Static edges — the spine of the graph
    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "route")

    # Conditional: route branches between two retrieval paths
    g.add_conditional_edges(
        "route",
        route_branch,
        {
            "simple": "retrieve",
            "multi_hop": "tool_use",
        },
    )

    # Both retrieval paths converge at rerank
    g.add_edge("retrieve", "rerank")
    g.add_edge("tool_use", "rerank")

    g.add_edge("rerank", "generate")
    g.add_edge("generate", "self_check")

    # Conditional: self_check either loops back or exits
    g.add_conditional_edges(
        "self_check",
        confidence_branch,
        {
            "retry": "rewrite_query",
            "done": END,
        },
    )

    return g.compile()


# Module-level compiled app — created once when this module is imported.
# Importers can do `from src.graph import app` and call app.invoke(...).
app = build_graph()


# ============================================================
# CLI for ad-hoc testing
# ============================================================

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "How is Helios's battery business growing?"

    print(f"Query: {query}\n")
    print("Running graph...\n")

    result = app.invoke({"query": query, "attempts": 0})

    print("=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.get("answer", "(no answer)"))
    print()
    print(f"Route taken:  {result.get('route', '?')}")
    print(f"Citations:    {result.get('citations', [])}")
    print(f"Confidence:   {result.get('confidence', 0.0):.2f}")
    print(f"Attempts:     {result.get('attempts', 0)}")
    print(f"Rewritten Q:  {result.get('rewritten', '?')}")
