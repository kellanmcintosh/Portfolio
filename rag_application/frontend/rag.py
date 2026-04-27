"""RAG pipeline: embedding, retrieval, and generation against Groq.

Pure logic — no Streamlit UI calls aside from `@st.cache_resource` on the
ChromaDB client (caching avoids reconnecting on every Streamlit rerun).
"""

import json
import logging

import chromadb
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import (
    CHAT_MODEL,
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    GROQ_API_KEY,
    GROQ_URL,
    REASONING_SUMMARY_PROMPT,
    SUMMARY_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
)

logger = logging.getLogger(__name__)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


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
    """Return (chunks, sources) for the top-K matches. Raises if the collection is empty."""
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
        f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks, strict=False)
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
    """Run a full non-streaming Q&A turn.

    Returns (answer, unique_sources, reasoning_summary, [(chunk, source), ...]).
    `reasoning_summary` is empty unless `include_reasoning=True` and the model
    actually emitted a <think> block.
    """
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
    return answer, unique_sources, reasoning, list(zip(chunks, sources, strict=False))


def stream_answer_tokens(
    question: str,
    chunks: list[str],
    sources: list[str],
    thinking_sink: list[str],
    system_prompt: str = SYSTEM_PROMPT,
):
    """Stream answer tokens from Groq, hiding the <think> block from the UI.

    Qwen3 emits its chain-of-thought wrapped in <think>...</think> at the start
    of the response. We need to display the answer tokens as they arrive but
    NOT the reasoning. The state machine below buffers tokens until it can
    decide whether the stream has entered <think> (drop until </think>, append
    raw text to `thinking_sink` for later summarization) or skipped it (yield
    everything from here on). The 15-char / no-`<` heuristic decides early
    enough to feel responsive without risking a half-matched `<think>` tag.
    """
    user_content = build_prompt(question, chunks, sources) if chunks else question
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
