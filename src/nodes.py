"""Graph nodes — the six functions that make up the RAG pipeline.

Each node is a pure function: takes RAGState, returns a partial dict
that LangGraph merges back into the state. No node mutates state directly.
"""
from pathlib import Path
import re

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_core.messages import HumanMessage

from src.state import RAGState

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
CONTEXTUAL_COLLECTION = "contextual"

# Model clients — instantiated once at module load.
# Cheap model (Haiku) for utility nodes, smart model (Sonnet) for generation.
_haiku = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)
_sonnet = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=1024)
_embeddings = CohereEmbeddings(model="embed-english-v3.0")
_reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
_vector_store = Chroma(
    collection_name=CONTEXTUAL_COLLECTION,
    embedding_function=_embeddings,
    persist_directory=str(CHROMA_DIR),
)


# ============================================================
# Node 1 — rewrite_query
# ============================================================
REWRITE_PROMPT = """You are a search query rewriter. Rewrite the user's query to improve retrieval against a corpus of company documents.

Rules:
- Expand acronyms and abbreviations.
- Add synonyms for ambiguous terms.
- Keep it under 30 words.
- Output ONLY the rewritten query, nothing else.

Query: {query}"""


def rewrite_query(state: RAGState) -> dict:
    """Expand and clarify the query for better retrieval."""
    response = _haiku.invoke([
        HumanMessage(content=REWRITE_PROMPT.format(query=state["query"]))
    ])
    return {"rewritten": response.content.strip()}


# ============================================================
# Node 2 — route
# ============================================================
ROUTE_PROMPT = """Classify whether this query needs:
- "simple": a single document lookup is enough
- "multi_hop": requires combining multiple docs, computation, or external tools

Output exactly one word: simple OR multi_hop.

Query: {query}"""


def route(state: RAGState) -> dict:
    """Classify the query as simple retrieval or multi-hop."""
    response = _haiku.invoke([
        HumanMessage(content=ROUTE_PROMPT.format(query=state["rewritten"]))
    ])
    decision = response.content.strip().lower()
    return {"route": "multi_hop" if "multi" in decision else "simple"}


# ============================================================
# Node 3 — retrieve
# ============================================================
def retrieve(state: RAGState) -> dict:
    """Pull top-20 candidate chunks from the contextual Chroma collection."""
    chunks = _vector_store.similarity_search(state["rewritten"], k=20)
    return {"chunks": chunks}


# ============================================================
# Node 4 — rerank
# ============================================================
def rerank(state: RAGState) -> dict:
    """Cohere reranker narrows top-20 candidates down to top-5 by relevance."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"reranked": []}

    results = _reranker.compress_documents(
        documents=chunks,
        query=state["rewritten"],
    )
    return {"reranked": results}


# ============================================================
# Node 5 — generate
# ============================================================
GENERATE_PROMPT = """You answer questions strictly using the provided context.

Rules:
- If the context doesn't contain the answer, say "I don't have enough information in the provided context to answer that."
- Cite source IDs in [brackets] inline as you make claims, e.g., [06_q3_2025_earnings].
- Be concise — under 150 words.
- Don't speculate beyond the context.

Context:
{context}

Question: {query}

Answer:"""


def generate(state: RAGState) -> dict:
    """Generate a grounded answer using the reranked chunks."""
    chunks = state.get("reranked", [])
    if not chunks:
        return {"answer": "No relevant context found.", "citations": []}

    context_text = "\n\n".join([
        f"[{c.metadata.get('doc_id', 'unknown')}]\n"
        f"{c.metadata.get('raw_chunk', c.page_content)}"
        for c in chunks
    ])

    response = _sonnet.invoke([
        HumanMessage(content=GENERATE_PROMPT.format(
            context=context_text,
            query=state["query"],
        ))
    ])

    citations = list(set(re.findall(r"\[([\w_]+)\]", response.content)))
    return {"answer": response.content.strip(), "citations": citations}


# ============================================================
# Node 6 — self_check
# ============================================================
SELF_CHECK_PROMPT = """You are a grounded-answer verifier. Given an answer and the chunks used to generate it, score how grounded the answer is.

Score 0.0-1.0:
- 1.0: every claim is directly supported by the context
- 0.5: some claims supported, others speculative
- 0.0: the answer contradicts or fabricates beyond the context

Output exactly one number between 0.0 and 1.0. No explanation, no other text.

Source chunks:
{context}

Answer:
{answer}

Score:"""


def self_check(state: RAGState) -> dict:
    """Verify the answer is grounded in retrieved chunks. Returns confidence and increments attempts."""
    chunks = state.get("reranked", [])
    answer = state.get("answer", "")
    attempts = state.get("attempts", 0) + 1

    if not chunks or not answer or "No relevant context" in answer:
        return {"confidence": 0.0, "attempts": attempts}

    context_text = "\n\n".join([c.page_content for c in chunks])

    response = _haiku.invoke([
        HumanMessage(content=SELF_CHECK_PROMPT.format(
            context=context_text,
            answer=answer,
        ))
    ])

    try:
        confidence = float(response.content.strip().split()[0])
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, IndexError):
        confidence = 0.5  # if parsing fails, assume middling confidence

    return {"confidence": confidence, "attempts": attempts}


# ============================================================
# Tool use — stub for multi_hop branch (Day 2 expansion)
# ============================================================
def tool_use(state: RAGState) -> dict:
    """Placeholder for multi-hop / tool-calling path.

    For now, falls back to standard retrieval. In production this would
    call SQL, web search, or domain-specific APIs based on the query.
    """
    return retrieve(state)
