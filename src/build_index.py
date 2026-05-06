"""Build two side-by-side Chroma collections — vanilla vs contextual."""
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.contextual import build_contextual_chunks

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
VANILLA = "vanilla"
CONTEXTUAL = "contextual"


def get_embeddings():
    return CohereEmbeddings(model="embed-english-v3.0")


def build_indices():
    chunks = build_contextual_chunks(use_cache=True)
    embeddings = get_embeddings()

    vanilla_docs = [
        Document(
            page_content=c.chunk_text,
            metadata={"doc_id": c.doc_id, "chunk_index": c.chunk_index},
        )
        for c in chunks
    ]
    contextual_docs = [
        Document(
            page_content=c.embedding_text,
            metadata={
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "raw_chunk": c.chunk_text,
            },
        )
        for c in chunks
    ]

    print(f"Building vanilla collection ({len(vanilla_docs)} chunks)...")
    Chroma.from_documents(
        documents=vanilla_docs,
        embedding=embeddings,
        collection_name=VANILLA,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Building contextual collection ({len(contextual_docs)} chunks)...")
    Chroma.from_documents(
        documents=contextual_docs,
        embedding=embeddings,
        collection_name=CONTEXTUAL,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"\nBoth collections persisted to {CHROMA_DIR}")


def compare(query, k=3):
    embeddings = get_embeddings()
    vanilla_store = Chroma(
        collection_name=VANILLA,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    contextual_store = Chroma(
        collection_name=CONTEXTUAL,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"\n{'=' * 70}")
    print(f"Query: {query}")
    print(f"{'=' * 70}")

    print(f"\n--- VANILLA retrieval (top {k}) ---")
    for i, doc in enumerate(vanilla_store.similarity_search(query, k=k), 1):
        print(f"  {i}. [{doc.metadata['doc_id']}]")
        print(f"     {doc.page_content[:130].strip()}...")

    print(f"\n--- CONTEXTUAL retrieval (top {k}) ---")
    for i, doc in enumerate(contextual_store.similarity_search(query, k=k), 1):
        raw = doc.metadata.get("raw_chunk", doc.page_content)
        print(f"  {i}. [{doc.metadata['doc_id']}]")
        print(f"     {raw[:130].strip()}...")


if __name__ == "__main__":
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        build_indices()
    else:
        print(f"(reusing existing indices at {CHROMA_DIR})\n")

    test_queries = [
        "How is Helios's battery business growing?",
        "What was the revenue trend in 2025?",
        "What financing options are available?",
        "How does the warranty work?",
    ]
    for q in test_queries:
        compare(q)
