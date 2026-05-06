"""RAGState — the schema of the data that flows through the LangGraph.

Every node reads this whole object and returns a partial dict that
gets merged into it. The graph is essentially a series of transformations
on this state.
"""
from typing import TypedDict, List, Literal, Optional
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    # Input
    query: str                              # the user's original question

    # Set by rewrite_query node
    rewritten: str                          # query expanded for retrieval

    # Set by route node
    route: Literal["simple", "multi_hop"]   # which retrieval path to take

    # Set by retrieve / tool_use node
    chunks: List[Document]                  # candidate chunks (top-K from vector search)

    # Set by rerank node
    reranked: List[Document]                # final chunks after Cohere rerank

    # Set by generate node
    answer: str                             # the LLM's grounded answer
    citations: List[str]                    # which chunk doc_ids were cited

    # Set by self_check node
    confidence: float                       # 0.0-1.0, is the answer grounded?
    attempts: int                           # how many retry loops have we taken

    # Optional, for tool_use multi-hop path
    tool_calls: Optional[List[dict]]
