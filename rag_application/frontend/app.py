import logging
from datetime import datetime

import streamlit as st

from rag import get_collection
from ui.components import render_assistant_turn, render_message, render_sidebar
from ui.styles import (
    apply_global_styles,
    inject_mode_badge,
    inject_width_fix,
    load_static,
    scroll_to_latest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

st.set_page_config(
    page_title="Internal Knowledge Retrieval Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
inject_width_fix()

show_reasoning = render_sidebar()

badge_label = "Strict Mode 🔒" if st.session_state.strict_mode else "Free Mode 🔓"
inject_mode_badge(badge_label)

collection = get_collection()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hero orb — only on empty state so it doesn't push chat down
if not st.session_state.messages:
    st.components.v1.html(load_static("orb.html"), height=560)

for msg in st.session_state.messages:
    render_message(msg, show_reasoning)

if question := st.chat_input("Ask your knowledge base…"):
    user_msg = {
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%I:%M %p"),
        "sources": [],
        "reasoning": "",
        "chunk_sources": [],
    }
    st.session_state.messages.append(user_msg)
    render_message(user_msg, show_reasoning)

    assistant_msg = render_assistant_turn(collection, question, show_reasoning)
    st.session_state.messages.append(assistant_msg)

    scroll_to_latest()
