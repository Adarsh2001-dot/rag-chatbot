"""
RAG Chatbot — Streamlit web UI
Run: streamlit run src/app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Chatbot")
    st.caption("Ask questions about any document")
    st.divider()

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.subheader("Settings")
    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 200, 50, step=10)
    top_k = st.slider("Chunks to retrieve (k)", 1, 6, 3)
    model = st.selectbox("LLM model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])

    st.divider()
    st.caption("Built with LangChain + FAISS + OpenAI")

# ── Main area ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

# Left column: document upload + chunk viewer
with col1:
    st.subheader("Document")
    uploaded = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])

    if uploaded and api_key:
        if st.button("Index document", type="primary"):
            with st.spinner(f"Chunking and embedding..."):
                # Save upload to temp file
                suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded.read())
                    tmp_path = f.name

                rag = RAGPipeline(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    model=model,
                )
                n = rag.load_document(tmp_path)
                rag.build_index()

                st.session_state["rag"] = rag
                st.session_state["chunks"] = rag.chunks
                st.session_state["messages"] = []
                st.session_state["hit_indices"] = []

        if "chunks" in st.session_state:
            chunks = st.session_state["chunks"]
            hits = st.session_state.get("hit_indices", [])

            st.caption(f"{len(chunks)} chunks indexed")
            c1, c2 = st.columns(2)
            c1.metric("Chunks", len(chunks))
            avg = sum(len(c.page_content) for c in chunks) // len(chunks)
            c2.metric("Avg chunk size", f"{avg} chars")

            st.caption("Chunks — green = last retrieved")
            for i, chunk in enumerate(chunks):
                color = "🟢" if i in hits else "⬜"
                with st.expander(f"{color} Chunk {i+1} — {chunk.page_content[:50]}..."):
                    st.text(chunk.page_content)
                    st.caption(f"Source: {chunk.metadata.get('source','?')} | Page: {chunk.metadata.get('page','?')}")

    elif not api_key:
        st.info("Enter your OpenAI API key in the sidebar to get started.")

# Right column: chat
with col2:
    st.subheader("Chat")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources retrieved"):
                    for src in msg["sources"]:
                        st.caption(f"Page {src['page']}: {src['preview']}")

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        if "rag" not in st.session_state:
            st.error("Upload and index a document first.")
        else:
            # Show user message
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    rag: RAGPipeline = st.session_state["rag"]
                    result = rag.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    source_docs = result["source_documents"]

                    # Track which chunks were hit
                    hit_idx = []
                    for doc in source_docs:
                        for i, chunk in enumerate(st.session_state["chunks"]):
                            if chunk.page_content == doc.page_content:
                                hit_idx.append(i)
                    st.session_state["hit_indices"] = hit_idx

                    st.markdown(answer)

                    sources = [
                        {
                            "page": d.metadata.get("page", "?"),
                            "preview": d.page_content[:100],
                        }
                        for d in source_docs
                    ]
                    with st.expander(f"Sources ({len(sources)} chunks retrieved)"):
                        for s in sources:
                            st.caption(f"Page {s['page']}: {s['preview']}...")

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
