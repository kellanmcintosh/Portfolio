"""Streamlit UI components: chat messages, sidebar, and the assistant turn.

Each function renders into the current Streamlit context and (where useful)
returns the data the entry point needs — e.g. the sidebar's `show_reasoning`
flag, or the assistant message dict to append to history.
"""

import logging
from datetime import datetime

import streamlit as st

from config import FREE_SYSTEM_PROMPT, SYSTEM_PROMPT
from rag import retrieve, stream_answer_tokens, summarize_thinking
from ui.styles import load_static

logger = logging.getLogger(__name__)


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


def render_sidebar() -> bool:
    """Render the sidebar; return show_reasoning flag."""
    if "strict_mode" not in st.session_state:
        st.session_state["strict_mode"] = True

    with st.sidebar:
        st.markdown(
            """
            <div style="padding:1.4rem 0 0.8rem;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
                <div style="width:26px;height:26px;border-radius:50%;
                            background:linear-gradient(135deg,#7C3AED,#A855F7);
                            box-shadow:0 0 14px rgba(168,85,247,0.55);flex-shrink:0;"></div>
                <span style="font-size:0.92rem;font-weight:600;color:#E2E8F0;letter-spacing:-0.01em;">
                  Knowledge Agent
                </span>
              </div>
              <p style="font-size:0.76rem;color:rgba(100,116,139,0.85);margin:0;line-height:1.55;">
                Grounded answers from your internal document library.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            """
            <p style="font-size:0.72rem;font-weight:600;color:#4B5563;
                      text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.65rem;">
              How to use
            </p>
            <ul style="font-size:0.79rem;color:#64748B;padding-left:1.1rem;
                       line-height:1.85;margin:0 0 0.5rem;">
              <li>Ask a question in plain English</li>
              <li>Answers are grounded in your documents</li>
              <li>Toggle Thought Train to see reasoning</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        show_reasoning = st.toggle("Thought Train", value=False)
        msg_count = len(st.session_state.get("messages", []))
        st.markdown(
            f'<p style="font-size:0.76rem;color:#374151;margin:0.6rem 0 0.8rem;">'
            f'{msg_count} message{"s" if msg_count != 1 else ""} this session</p>',
            unsafe_allow_html=True,
        )
        st.divider()
        mode_label = "Strict Mode 🔒" if st.session_state.strict_mode else "Free Mode 🔓"
        strict_mode = st.toggle(mode_label, key="strict_mode")
        mode_desc = (
            "Answers limited to the knowledge base"
            if strict_mode
            else "Answers anything, knowledge base not enforced"
        )
        st.markdown(
            f'<p style="font-size:0.72rem;color:rgba(100,116,139,0.7);margin:0.25rem 0 0.5rem;">'
            f"{mode_desc}</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return show_reasoning


def render_assistant_turn(collection, question: str, show_reasoning: bool) -> dict:
    """Run retrieval + streaming + reasoning summary, render to UI, return the assistant message dict."""
    answer = ""
    sources: list[str] = []
    reasoning = ""
    chunk_sources: list[tuple[str, str]] = []

    with st.chat_message("assistant"):
        response_slot = st.empty()
        try:
            if st.session_state.get("strict_mode", True):
                with st.spinner("Searching knowledge base…"):
                    retrieved_chunks, retrieved_sources = retrieve(collection, question)
                active_prompt = SYSTEM_PROMPT
            else:
                retrieved_chunks, retrieved_sources = [], []
                active_prompt = FREE_SYSTEM_PROMPT

            thinking_sink: list[str] = []

            # Typing indicator occupies the slot until the first token arrives
            response_slot.markdown(load_static("typing.html"), unsafe_allow_html=True)

            full_answer = ""
            for token in stream_answer_tokens(
                question, retrieved_chunks, retrieved_sources, thinking_sink, active_prompt
            ):
                full_answer += token
                # Streaming cursor gives typewriter feel; markdown re-renders via React diff
                response_slot.markdown(full_answer + " ▌")

            response_slot.markdown(full_answer if full_answer else "*(no response)*")
            answer = full_answer

            sources = list(dict.fromkeys(retrieved_sources))
            chunk_sources = list(zip(retrieved_chunks, retrieved_sources, strict=False))

            if show_reasoning and thinking_sink:
                with st.spinner("Summarizing reasoning…"):
                    reasoning = summarize_thinking("".join(thinking_sink))

        except Exception as e:
            logger.exception("Failed to answer question: %.80s", question)
            answer = f"Something went wrong: {e}"
            response_slot.error(answer)

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

    return {
        "role": "assistant",
        "content": answer,
        "timestamp": assistant_ts,
        "sources": sources,
        "reasoning": reasoning,
        "chunk_sources": chunk_sources,
    }
