import json
import logging
import os
from datetime import datetime

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

CSS = """
<style>
/* ── Layout ───────────────────────────────────────────────────────── */
.main .block-container {
    max-width: 820px;
    padding-top: 1.5rem;
    padding-bottom: 6rem;
}

/* ── Fade-in animation ────────────────────────────────────────────── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stChatMessage"] {
    animation: fadeInUp 0.22s ease-out;
    border-radius: 14px;
    padding: 6px 10px;
    margin-bottom: 6px;
}

/* ── User bubble ──────────────────────────────────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #EEF2FF;
    border-left: 3px solid #6366F1;
    margin-left: 10%;
}

/* ── Assistant bubble ─────────────────────────────────────────────── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
    margin-right: 10%;
}

/* ── Chat input ───────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 2px solid #E5E7EB !important;
    padding: 12px 20px !important;
    font-size: 0.95rem !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    resize: none !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    outline: none !important;
}

/* ── Timestamps ───────────────────────────────────────────────────── */
.msg-timestamp {
    font-size: 0.7rem;
    color: #9CA3AF;
    display: block;
    margin-top: 2px;
}

/* ── Sidebar ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F9FAFB;
    color: #1F2937;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown {
    color: #1F2937 !important;
}

/* ── Clear button ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stButton button {
    border: 1.5px solid #E5E7EB;
    background: white;
    color: #374151;
    border-radius: 8px;
    font-size: 0.88rem;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: #EF4444;
    color: #EF4444;
    background: #FEF2F2;
}

/* ── Mobile ───────────────────────────────────────────────────────── */
@media (max-width: 640px) {
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-left: 0;
        margin-right: 0;
    }
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
"""

SCROLL_JS = """
<script>
(function() {
    var msgs = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
    if (msgs.length) msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
})();
</script>
"""


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
    """Split a Qwen3 response into (thinking, answer)."""
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


def stream_answer_tokens(
    question: str,
    chunks: list[str],
    sources: list[str],
    thinking_sink: list[str],
):
    """Yields answer tokens for st.write_stream, buffering the <think> block silently."""
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(question, chunks, sources)},
            ],
            "stream": True,
        },
        stream=True,
    )
    if not response.ok:
        logger.error("Groq stream failed: %s %s", response.status_code, response.reason)
        raise Exception(f"{response.status_code} {response.reason}: {response.json()}")

    buffer = ""
    in_think = False
    think_done = False

    for line in response.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        if payload == "[DONE]":
            break
        try:
            token = json.loads(payload)["choices"][0]["delta"].get("content", "")
        except (KeyError, json.JSONDecodeError):
            continue
        if not token:
            continue

        buffer += token

        if not in_think and not think_done:
            if "<think>" in buffer:
                in_think = True
                buffer = buffer.split("<think>", 1)[1]
            elif len(buffer) > 15 and "<" not in buffer:
                think_done = True
                yield buffer
                buffer = ""
        elif in_think:
            if "</think>" in buffer:
                think_text, answer_part = buffer.split("</think>", 1)
                thinking_sink.append(think_text)
                buffer = answer_part.lstrip("\n")
                in_think = False
                think_done = True
                if buffer:
                    yield buffer
                    buffer = ""
        else:
            yield buffer
            buffer = ""

    if buffer:
        yield buffer


def render_message(msg: dict, show_reasoning: bool) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("timestamp"):
            st.markdown(
                f'<span class="msg-timestamp">{msg["timestamp"]}</span>',
                unsafe_allow_html=True,
            )
        if show_reasoning and msg.get("reasoning"):
            with st.expander("Thought Train"):
                st.markdown(msg["reasoning"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))
        for i, (chunk, source) in enumerate(msg.get("chunk_sources", []), 1):
            with st.expander(f"Excerpt {i} — {source}"):
                st.write(chunk)


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Document Q&A", page_icon="💬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("Document Q&A")
    st.caption("Ask questions about your documents, powered by AI.")
    st.divider()
    show_reasoning = st.toggle("Thought Train")
    msg_count = len(st.session_state.get("messages", []))
    st.caption(f"{msg_count} message{'s' if msg_count != 1 else ''} in this session")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

collection = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; padding:4rem 0; color:#9CA3AF;">
            <div style="font-size:2.8rem; margin-bottom:0.75rem;">💬</div>
            <p style="font-size:1.05rem; font-weight:600; color:#6B7280; margin:0;">
                Ask a question to get started
            </p>
            <p style="font-size:0.85rem; margin-top:0.4rem;">
                Your answers are grounded in the documents you have uploaded.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    render_message(msg, show_reasoning)

if question := st.chat_input("Ask a question about your documents…"):
    user_ts = datetime.now().strftime("%I:%M %p")
    user_msg = {
        "role": "user",
        "content": question,
        "timestamp": user_ts,
        "sources": [],
        "reasoning": "",
        "chunk_sources": [],
    }
    st.session_state.messages.append(user_msg)
    render_message(user_msg, show_reasoning)

    answer = ""
    sources = []
    reasoning = ""
    chunk_sources = []

    with st.chat_message("assistant"):
        try:
            with st.spinner("Finding relevant context…"):
                retrieved_chunks, retrieved_sources = retrieve(collection, question)

            thinking_sink: list[str] = []
            answer = st.write_stream(
                stream_answer_tokens(question, retrieved_chunks, retrieved_sources, thinking_sink)
            )

            sources = list(dict.fromkeys(retrieved_sources))
            chunk_sources = list(zip(retrieved_chunks, retrieved_sources))

            if show_reasoning and thinking_sink:
                with st.spinner("Summarizing reasoning…"):
                    reasoning = summarize_thinking("".join(thinking_sink))

        except Exception as e:
            logger.exception("Failed to answer question: %.80s", question)
            answer = f"Something went wrong: {e}"
            st.error(answer)

        assistant_ts = datetime.now().strftime("%I:%M %p")
        st.markdown(
            f'<span class="msg-timestamp">{assistant_ts}</span>',
            unsafe_allow_html=True,
        )
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
        "timestamp": assistant_ts,
        "sources": sources,
        "reasoning": reasoning,
        "chunk_sources": chunk_sources,
    })

    st.markdown(SCROLL_JS, unsafe_allow_html=True)
