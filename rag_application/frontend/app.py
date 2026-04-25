import os
import chromadb
import google.generativeai as genai
import streamlit as st

COLLECTION_NAME = "documents"
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-1.5-pro"
TOP_K = 5

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_collection():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client.get_or_create_collection(COLLECTION_NAME)


def embed_query(text: str) -> list[float]:
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query",
    )
    return result["embedding"]


def retrieve(collection, question: str) -> tuple[list[str], list[str]]:
    query_embedding = embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def build_prompt(question: str, chunks: list[str], sources: list[str]) -> str:
    context_blocks = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
    )
    return (
        "You are a helpful assistant. Answer the question using only the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def ask(collection, question: str) -> tuple[str, list[str]]:
    chunks, sources = retrieve(collection, question)
    prompt = build_prompt(question, chunks, sources)
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    unique_sources = list(dict.fromkeys(sources))
    return response.text, unique_sources


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Document Q&A")

collection = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))

if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                answer, sources = ask(collection, question)
            except Exception as e:
                answer = f"Something went wrong: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " · ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
