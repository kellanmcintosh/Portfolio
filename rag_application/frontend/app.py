import os
import requests
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "documents"
CHAT_MODEL = "gemini-2.0-flash"
TOP_K = 5

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only "
    "the context provided. If the answer is not in the context, say you don't know."
)


@st.cache_resource
def get_collection():
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chroma_client.get_or_create_collection(COLLECTION_NAME)


def embed_query(text: str) -> list[float]:
    return embed_model.encode(text, normalize_embeddings=True).tolist()


def retrieve(collection, question: str) -> tuple[list[str], list[str]]:
    results = collection.query(
        query_embeddings=[embed_query(question)],
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
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_blocks}\n\n"
        f"Question: {question}"
    )


def ask(collection, question: str) -> tuple[str, list[str]]:
    chunks, sources = retrieve(collection, question)
    url = f"{GEMINI_BASE}/models/{CHAT_MODEL}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": build_prompt(question, chunks, sources)}]}],
    }
    response = requests.post(url, params={"key": GEMINI_API_KEY}, json=body)
    if not response.ok:
        raise Exception(f"{response.status_code} {response.reason}: {response.json()}")
    answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    unique_sources = list(dict.fromkeys(sources))
    return answer, unique_sources


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
