"""Contextual retrieval: prepend LLM-generated context to each chunk before embedding.

Implements the technique from Anthropic's "Contextual Retrieval" post:
https://www.anthropic.com/news/contextual-retrieval

For each chunk, we ask Haiku to write a 1-2 sentence context that situates
the chunk within the full document. We embed (context + chunk) together,
which preserves semantic information that would otherwise be lost when
chunks are embedded in isolation.
"""
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
import json

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = Path(__file__).parent.parent / "data" / ".contextual_chunks_cache.json"


@dataclass
class ContextualChunk:
    doc_id: str
    chunk_index: int
    chunk_text: str           # raw chunk
    context: str              # LLM-generated context prefix
    embedding_text: str       # context + chunk, the thing we actually embed


CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def load_documents() -> dict:
    """Load all .md files from data/. Returns {doc_id: full_text}."""
    docs = {}
    for path in sorted(DATA_DIR.glob("*.md")):
        docs[path.stem] = path.read_text()
    return docs


def chunk_document(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
    """Split a document into overlapping chunks. Small chunks here so we get
    multiple per doc — useful for showing contextual retrieval winning."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def generate_context(document: str, chunk: str, llm: ChatAnthropic) -> str:
    """Ask Haiku to write a 1-2 sentence context prefix for this chunk."""
    prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def build_contextual_chunks(use_cache: bool = True) -> List[ContextualChunk]:
    """Build contextual chunks for the entire corpus.
    
    Caches results to disk so we don't re-pay for context generation
    on every run during development.
    """
    if use_cache and CACHE_PATH.exists():
        print(f"Loading cached contextual chunks from {CACHE_PATH.name}")
        data = json.loads(CACHE_PATH.read_text())
        return [ContextualChunk(**c) for c in data]
    
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=200)
    docs = load_documents()
    contextual_chunks: List[ContextualChunk] = []
    
    for doc_id, doc_text in docs.items():
        chunks = chunk_document(doc_text)
        for i, chunk in enumerate(chunks):
            print(f"  [{doc_id}] chunk {i+1}/{len(chunks)}...", end=" ", flush=True)
            context = generate_context(doc_text, chunk, llm)
            print("done")
            contextual_chunks.append(ContextualChunk(
                doc_id=doc_id,
                chunk_index=i,
                chunk_text=chunk,
                context=context,
                embedding_text=f"{context}\n\n{chunk}",
            ))
    
    # Cache results to avoid re-spending tokens on every run
    CACHE_PATH.write_text(json.dumps([asdict(c) for c in contextual_chunks], indent=2))
    print(f"\nCached {len(contextual_chunks)} chunks to {CACHE_PATH.name}")
    return contextual_chunks


if __name__ == "__main__":
    print("Building contextual chunks for Helios Solar corpus...\n")
    chunks = build_contextual_chunks(use_cache=False)
    
    print(f"\n=== Generated {len(chunks)} contextual chunks ===\n")
    print("--- Example: first chunk ---")
    print(f"Doc:     {chunks[0].doc_id}")
    print(f"Context: {chunks[0].context}")
    print(f"Chunk:   {chunks[0].chunk_text[:200]}...")
    print("\n--- Example: a chunk from earnings ---")
    earnings_chunk = next((c for c in chunks if "earnings" in c.doc_id), None)
    if earnings_chunk:
        print(f"Doc:     {earnings_chunk.doc_id}")
        print(f"Context: {earnings_chunk.context}")
        print(f"Chunk:   {earnings_chunk.chunk_text[:200]}...")
