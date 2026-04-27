# RAG Application

## Overview
Local document Q&A app. Users drop PDFs or other documents into `documents/`, run ingestion, then ask questions via a Streamlit chat interface. Answers are grounded in the documents with source attribution.

## Stack
- **Embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2`, runs fully local inside Docker, 384-dim vectors
- **Vector DB:** ChromaDB — persisted in a named Docker volume (`chroma_data`)
- **Generation:** Groq API — `qwen/qwen3-32b`, free tier (supports native chain-of-thought via `<think>` tags)
- **Document conversion:** Docling — converts PDF, Word, etc. to markdown
- **Chunking:** tiktoken `cl100k_base`, 500-token chunks, 50-token overlap
- **Frontend:** Streamlit on port 8501

## Project Structure
```
rag_application/
├── docker-compose.yml        3 services: chromadb, ingestion, frontend
├── .env                      GROQ_API_KEY, CHROMA_HOST, CHROMA_PORT (not committed)
├── .env.example              template
├── pyproject.toml            ruff config (first-party: ingest, app, config, rag, ui)
├── documents/                drop input files here (gitignored)
├── ingestion/
│   ├── ingest.py             convert → chunk → embed → upsert to ChromaDB
│   ├── Dockerfile            downloads all-MiniLM-L6-v2 at build time
│   └── requirements.txt
├── frontend/
│   ├── app.py                Streamlit entry point — page composition only (~60 LoC)
│   ├── config.py             env vars, model IDs, prompts
│   ├── rag.py                pipeline: get_collection, embed_query, retrieve,
│   │                         build_prompt, parse_response, summarize_thinking,
│   │                         ask, stream_answer_tokens — all tested via tests/test_app.py
│   ├── ui/
│   │   ├── components.py     render_message, render_sidebar, render_assistant_turn
│   │   └── styles.py         load_static, apply_global_styles, inject_width_fix,
│   │                         inject_mode_badge, scroll_to_latest
│   ├── static/
│   │   ├── styles.css        global Streamlit overrides + dark theme
│   │   ├── orb.html          self-contained iframe doc for the hero animation
│   │   └── typing.html       three-dot typing indicator shown while streaming
│   ├── Dockerfile            downloads all-MiniLM-L6-v2 at build time
│   └── requirements.txt
├── scripts/
│   ├── start.command         docker compose up --build chromadb frontend + open browser
│   ├── add-documents.command docker compose run --build --rm ingestion
│   └── stop.command          docker compose down
└── tests/
    ├── conftest.py           mocks sentence_transformers + streamlit, sets env vars
    ├── test_ingest.py        chunk_text + embed unit tests
    └── test_app.py           imports `rag` — build_prompt, embed_query, retrieve, ask, etc.
```

### Where to put new code

| Adding… | Goes in |
|---|---|
| A new env var or prompt | `frontend/config.py` |
| A new step in the RAG pipeline | `frontend/rag.py` (and a test in `tests/test_app.py`) |
| A new chat / sidebar element | `frontend/ui/components.py` |
| A CSS rule or HTML asset | `frontend/static/` (loader in `ui/styles.py`) |
| A new ingestion step | `ingestion/ingest.py` (and a test in `tests/test_ingest.py`) |

`app.py` should stay thin. If you find yourself adding business logic or large markup blocks there, it belongs in `rag.py`, `ui/`, or `static/`.

## How to Run
1. Copy `.env.example` → `.env`, add `GROQ_API_KEY`
2. Drop documents into `documents/`
3. Double-click `scripts/start.command`
4. Double-click `scripts/add-documents.command` to ingest
5. Ask questions at `http://localhost:8501`

**Local dev commands:**
```bash
pip install -r requirements-test.txt
pytest                  # run tests
ruff check .            # lint
ruff format .           # format
```

## Key Decisions
- `--build` on all scripts — prevents stale Docker image issues after code changes
- `upsert` not `add` in ChromaDB — re-running ingestion is idempotent
- `min(TOP_K, collection.count())` in retrieve — prevents ChromaDB error on small collections
- Embedding model baked into Docker image at build time — no network call at container startup
- Groq chosen over Gemini — Gemini quota limits (limit: 0) made it unusable on free tier

---

## Coding Standards

### Linter — ruff
All code must pass `ruff check .` before committing. Config lives in `pyproject.toml`.

Rules enforced:
- `E` / `W` — pycodestyle errors and warnings
- `F` — pyflakes (unused imports, undefined names)
- `I` — isort (import ordering)
- `UP` — pyupgrade (modern Python syntax)
- `B` — flake8-bugbear (common bugs and design issues)

Run before every commit:
```bash
ruff check .        # check for violations
ruff format .       # auto-format (replaces black)
```

Never suppress a ruff rule with `# noqa` without a comment explaining why.

---

### Logging
Use Python's `logging` module throughout. Never use `print()` in production code — `print()` is only acceptable in the ingestion container where output is piped to the user's terminal by the shell script.

**Standard setup — put this at the top of every module:**
```python
import logging
logger = logging.getLogger(__name__)
```

**Log level guide:**
| Level | When to use |
|---|---|
| `DEBUG` | Detailed internal state — chunk counts, embedding dimensions, timing |
| `INFO` | Normal progress milestones — file converted, chunks upserted, query received |
| `WARNING` | Recoverable issues — file skipped, empty result set, slow response |
| `ERROR` | Failures that affect the user — API call failed, ChromaDB unreachable |

**Rules:**
- Log at the start and end of every major operation (convert, chunk, embed, upsert, retrieve, generate)
- Always log the exception in except blocks: `logger.exception("message")` not `logger.error("message")`
- Never log secrets — no API keys, no full request bodies that may contain keys
- Include useful context in messages: filenames, counts, durations — not just "done"

**Example:**
```python
logger.info("Converting %d file(s)", len(files))
logger.debug("Chunked %s into %d chunks", file_path.name, len(chunks))
logger.warning("Skipping %s: %s", file_path.name, e)
logger.error("ChromaDB upsert failed for %s", file_path.name)
```

---

### Testing
Run with: `pytest` from `rag_application/`

**What requires a test:**
- Every pure function (no I/O, no external calls) — test all meaningful input variants
- Every function that calls an external API — mock the call, assert correct arguments and response parsing
- Every error/edge case that has burned us in production (empty collection, missing env var, bad API response)

**What does not require a test:**
- Streamlit UI rendering code
- Docker / shell scripts
- `if __name__ == "__main__"` entrypoints
- One-line wrappers that just delegate to a well-tested library

**Test naming:**
```
test_<function>_<scenario>
test_chunk_text_empty_input
test_embed_returns_one_vector_per_text
test_retrieve_raises_when_collection_empty
```

**Mock boundary:** Mock at the network edge only — never mock internal functions of the module under test. If a function is worth testing, test it directly.

**Patch targets:** RAG functions live in `rag.py`, so tests patch `rag.requests.post`, `rag.embed_query`, etc. — never `app.*`. `app.py` is a Streamlit entry point, not a place tests reach into.

---

### Coding Guidelines

#### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

#### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

#### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.
- Every changed line should trace directly to the user's request.

#### 4. Committing
Commit when the user explicitly asks, or suggest a commit when a meaningful chunk of work is complete — not after every small change. Batching related edits into one commit keeps the history readable. Always stage specific files by name.

#### 5. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan with verify steps before starting.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
