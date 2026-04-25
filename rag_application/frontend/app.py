import logging
import os

import requests
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
CHAT_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only "
    "the context provided. If the answer is not in the context, say you don't know."
)


@st.cache_resource
def get_collection():
    logger.info("Connecting to ChromaDB at %s:%s", CHROMA_HOST, CHROMA_PORT)
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    logger.info("Collection '%s' loaded — %d chunk(s) stored", COLLECTION_NAME, collection.count())
    return collection


def embed_query(text: str) -> list[float]:
    logger.debug("Embedding query: %.80s", text)
    return embed_model.encode(text, normalize_embeddings=True).tolist()


def retrieve(collection, question: str) -> tuple[list[str], list[str]]:
    count = collection.count()
    if count == 0:
        raise ValueError(
            "No documents have been ingested yet. "
            "Add files to the documents/ folder and run add-documents.command first."
        )
    n = min(TOP_K, count)
    logger.info("Retrieving top %d chunk(s) for query: %.80s", n, question)
    results = collection.query(
        query_embeddings=[embed_query(question)],
        n_results=n,
        include=["documents", "metadatas"],
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    logger.debug("Retrieved %d chunk(s) from source(s): %s", len(chunks), set(sources))
    return chunks, sources


def build_prompt(question: str, chunks: list[str], sources: list[str]) -> str:
    context_blocks = "\n\n".join(
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
    )
    return f"Context:\n{context_blocks}\n\nQuestion: {question}"


def ask(collection, question: str) -> tuple[str, list[str]]:
    chunks, sources = retrieve(collection, question)
    logger.info("Calling Groq (%s)", CHAT_MODEL)
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks, sources)},
            ],
        },
    )
    if not response.ok:
        logger.error("Groq request failed: %s %s", response.status_code, response.reason)
        raise Exception(f"{response.status_code} {response.reason}: {response.json()}")
    answer = response.json()["choices"][0]["message"]["content"]
    unique_sources = list(dict.fromkeys(sources))
    logger.info("Answer received — sources: %s", unique_sources)
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
                logger.exception("Failed to answer question: %.80s", question)
                answer = f"Something went wrong: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " · ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
