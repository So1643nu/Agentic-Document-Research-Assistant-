"""Streamlit UI for the Agentic RAG Research Assistant.

Run with: streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.agent import AgenticRAGAgent
from src.config import load_settings
from src.ingest import ingest_directory, ingest_uploaded_files
from src.vectorstore import build_store

SAMPLE_DOCS_DIR = Path(__file__).parent / "data" / "sample_docs"

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner=False)
def get_settings():
    return load_settings()


def init_state():
    if "store" not in st.session_state:
        settings = get_settings()
        st.session_state.store = build_store(use_embeddings=settings.use_embeddings)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # UI display history
    if "history" not in st.session_state:
        st.session_state.history = []  # API-format history for the agent
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []


def build_index_from_sample():
    settings = get_settings()
    chunks = ingest_directory(
        SAMPLE_DOCS_DIR,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    st.session_state.store.add_documents(chunks)
    st.session_state.indexed_files = sorted({c.source for c in chunks})


def build_index_from_uploads(uploaded_files):
    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [(f.name, f.getvalue()) for f in uploaded_files]
        chunks = ingest_uploaded_files(
            pairs,
            tmp_dir=tmp,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    st.session_state.store.add_documents(chunks)
    st.session_state.indexed_files = sorted(
        set(st.session_state.indexed_files) | {c.source for c in chunks}
    )


def main():
    init_state()
    settings = get_settings()

    st.title("🔎 Agentic RAG Research Assistant")
    st.caption(
        "The model decides on its own whether to search your documents, "
        "run a calculation, or just answer — it isn't forced through a "
        "fixed retrieve-then-generate pipeline."
    )

    with st.sidebar:
        st.header("Knowledge base")

        if not settings.anthropic_api_key:
            st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

        backend = "Embeddings (FAISS)" if settings.use_embeddings else "TF-IDF (offline default)"
        st.caption(f"Retrieval backend: **{backend}**")

        if st.button("📚 Build index from sample docs", use_container_width=True):
            with st.spinner("Ingesting sample documents..."):
                build_index_from_sample()
            st.success(f"Indexed {len(st.session_state.indexed_files)} file(s).")

        uploaded = st.file_uploader(
            "Or upload your own documents",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("📥 Index uploaded files", use_container_width=True):
            with st.spinner("Ingesting uploaded documents..."):
                build_index_from_uploads(uploaded)
            st.success(f"Indexed {len(st.session_state.indexed_files)} file(s) total.")

        if st.session_state.indexed_files:
            st.write("**Indexed files:**")
            for f in st.session_state.indexed_files:
                st.write(f"- {f}")
        else:
            st.info("No documents indexed yet.")

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander("🔧 Tool calls made by the agent"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**{tc['tool_name']}**  \nInput: `{tc['tool_input']}`")
                        st.code(tc["result"][:1500])

    user_query = st.chat_input("Ask a question...")
    if user_query:
        if not settings.anthropic_api_key:
            st.error("Set ANTHROPIC_API_KEY in your .env file before chatting.")
            return

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        agent = AgenticRAGAgent(settings=settings, store=st.session_state.store)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.run(user_query, history=st.session_state.history)
            st.markdown(response.text)
            if response.tool_calls:
                with st.expander("🔧 Tool calls made by the agent"):
                    for tc in response.tool_calls:
                        st.markdown(f"**{tc.tool_name}**  \nInput: `{tc.tool_input}`")
                        st.code(tc.result[:1500])

        st.session_state.history.append({"role": "user", "content": user_query})
        st.session_state.history.append({"role": "assistant", "content": response.text})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"tool_name": tc.tool_name, "tool_input": tc.tool_input, "result": tc.result}
                    for tc in response.tool_calls
                ],
            }
        )


if __name__ == "__main__":
    main()
