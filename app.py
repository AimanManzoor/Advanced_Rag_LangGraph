"""Streamlit demo for Helios RAG — self-correcting RAG with LangGraph."""
import streamlit as st
from src.graph import app

st.set_page_config(page_title="Helios RAG", page_icon="☀️", layout="wide")
st.title("☀️ Helios RAG — Self-Correcting Retrieval-Augmented Generation")
st.caption("LangGraph + Anthropic Claude + Cohere · 100% citation accuracy on a 30-question eval")

with st.sidebar:
    st.header("About this demo")
    st.markdown("""
    A self-correcting RAG pipeline over 10 documents about a fictional 
    solar company. Ask anything about Helios Solar's products, financing, 
    warranties, earnings, or customer stories.
    
    **Architecture:** query rewriter → router → vector retrieval → Cohere rerank → 
    Sonnet generation → Haiku self-check (with retry loop if confidence < 0.7).
    
    [GitHub repo](https://github.com/AimanManzoor/Advanced_Rag_LangGraph) · 
    [Architecture doc](https://github.com/AimanManzoor/Advanced_Rag_LangGraph/blob/main/docs/architecture.md)
    """)
    
    st.divider()
    st.subheader("Try these")
    examples = [
        "How is Helios's battery business growing?",
        "What was the revenue trend in 2025?",
        "What financing options are available?",
        "What is the meaning of life?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["query"] = ex

query = st.text_input(
    "Your question:",
    value=st.session_state.get("query", ""),
    placeholder="Ask anything about Helios Solar...",
)

if st.button("Run query", type="primary") and query:
    with st.spinner("Running the graph..."):
        result = app.invoke({"query": query, "attempts": 0})

    st.markdown("### Answer")
    st.write(result.get("answer", "(no answer)"))

    cols = st.columns(4)
    cols[0].metric("Route", result.get("route", "?"))
    cols[1].metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
    cols[2].metric("Attempts", result.get("attempts", 0))
    cols[3].metric("Citations", len(result.get("citations", [])))

    if result.get("citations"):
        st.markdown("### Sources cited")
        for c in result["citations"]:
            st.markdown(f"- `{c}`")

    with st.expander("Show rewritten query and reranked chunks"):
        st.write("**Rewritten query:**", result.get("rewritten", "?"))
        st.write("**Reranked chunks:**")
        for i, doc in enumerate(result.get("reranked", []), 1):
            doc_id = doc.metadata.get("doc_id", "?")
            content = doc.metadata.get("raw_chunk", doc.page_content)
            st.markdown(f"**{i}. [{doc_id}]**")
            st.text(content[:300] + "...")
