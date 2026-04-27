"""Frontend configuration: environment variables, model IDs, and prompts.

Imported eagerly so missing env vars fail loudly at startup rather than on
the first user query.
"""

import os

COLLECTION_NAME = "documents"
CHAT_MODEL = "qwen/qwen3-32b"
SUMMARY_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only "
    "the context provided. If the answer is not in the context, say you don't know."
)

FREE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question freely using your full capabilities."
)

REASONING_SUMMARY_PROMPT = (
    "The text below is the internal reasoning a language model used to answer a question. "
    "Rewrite it in 2-3 clear sentences that anyone can understand, without technical jargon. "
    "Explain what information was used and why the model reached its conclusion."
)
