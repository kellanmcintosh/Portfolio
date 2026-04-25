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
CHAT_MODEL = "qwen/qwen3-32b"
SUMMARY_MODEL = "llama-3.1-8b-instant"
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

REASONING_SUMMARY_PROMPT = (
    "The text below is the internal reasoning a language model used to answer a question. "
    "Rewrite it in 2-3 clear sentences that anyone can understand, without technical jargon. "
    "Explain what information was used and why the model reached its conclusion."
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


def parse_response(content: str) -> tuple[str, str]:
    """Split a DeepSeek-R1 response into (thinking, answer)."""
    if "<think>" in content and "</think>" in content:
        thinking = content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
        answer = content.split("</think>", 1)[1].strip()
        return thinking, answer
    return "", content


def summarize_thinking(thinking: str) -> str:
    logger.info("Summarizing model reasoning (%d chars)", len(thinking))
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": SUMMARY_MODEL,
            "messages": [
                {"role": "system", "content": REASONING_SUMMARY_PROMPT},
                {"role": "user", "content": thinking},
            ],
        },
    )
    if not response.ok:
        logger.warning("Reasoning summarization failed: %s", response.status_code)
        return ""
    return response.json()["choices"][0]["message"]["content"]


def ask(
    collection, question: str, include_reasoning: bool = False
) -> tuple[str, list[str], str, list[tuple[str, str]]]:
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
    content = response.json()["choices"][0]["message"]["content"]
    raw_thinking, answer = parse_response(content)
    reasoning = summarize_thinking(raw_thinking) if include_reasoning and raw_thinking else ""
    unique_sources = list(dict.fromkeys(sources))
    logger.info("Answer received — sources: %s", unique_sources)
    return answer, unique_sources, reasoning, list(zip(chunks, sources))


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chat", page_icon="📄")
st.title("📄 Document Q&A")

show_reasoning = st.sidebar.toggle("Thought Train")

collection = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_reasoning and msg.get("reasoning"):
            with st.expander("Thought Train"):
                st.markdown(msg["reasoning"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))
        for i, (chunk, source) in enumerate(msg.get("chunk_sources", []), 1):
            with st.expander(f"Excerpt {i} — {source}"):
                st.write(chunk)

if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                answer, sources, reasoning, chunk_sources = ask(
                    collection, question, include_reasoning=show_reasoning
                )
            except Exception as e:
                logger.exception("Failed to answer question: %.80s", question)
                answer = f"Something went wrong: {e}"
                sources = []
                reasoning = ""
                chunk_sources = []
        st.markdown(answer)
        if show_reasoning and reasoning:
            with st.expander("Thought Train"):
                st.markdown(reasoning)
        if sources:
            st.caption("Sources: " + " · ".join(sources))
        for i, (chunk, source) in enumerate(chunk_sources, 1):
            with st.expander(f"Excerpt {i} — {source}"):
                st.write(chunk)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "reasoning": reasoning,
        "chunk_sources": chunk_sources,
    })
