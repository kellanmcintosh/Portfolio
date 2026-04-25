# Local RAG Application

A local document question-answering app. Drop in PDFs or Word docs, ask questions in a chat interface, and get answers grounded in your documents — all running on your machine.

Built with [Docling](https://github.com/DS4SD/docling), [ChromaDB](https://www.trychroma.com/), [Groq](https://console.groq.com/), and [Streamlit](https://streamlit.io/), orchestrated via Docker Compose.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A [Groq API key](https://console.groq.com/keys) (free tier works)

---

## First-time setup

**1. Copy the environment file and add your key**

```bash
cp .env.example .env
```

Open `.env` and replace `your_groq_api_key_here` with your real key:

```
GROQ_API_KEY=gsk_...
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

> `.env` is gitignored — your key never leaves your machine.

**2. Add your documents**

Copy any PDFs, Word docs, or text files into the `documents/` folder.

**3. Start the app**

Double-click `scripts/start.command`.

> The first run will download and build Docker images — this takes a few minutes. Subsequent starts are fast.

**4. Ingest your documents**

Double-click `scripts/add-documents.command`.

This converts your documents to text, chunks them, and loads them into the local vector database. You only need to run this when you add new documents.

**5. Ask questions**

The browser opens automatically at `http://localhost:8501`. Type a question and get an answer with source references.

---

## Features

- **Thought Train** — toggle in the sidebar to see a plain-English summary of how the model reasoned through your question
- **Source excerpts** — collapsible excerpts below each answer show the exact text the model used

---

## Daily use

| Task | Script |
|---|---|
| Start the app | `scripts/start.command` |
| Add new documents | `scripts/add-documents.command` |
| Stop the app | `scripts/stop.command` |

---

## How it works

```
documents/          ← you drop files here
     │
     ▼
 ingestion          converts docs → markdown → 500-token chunks
     │              embeds each chunk locally with sentence-transformers
     ▼
 ChromaDB           stores vectors + raw text locally on your machine
     │
     ▼
 frontend           embeds your question, finds the 5 closest chunks,
                    sends them as context to Qwen3-32b via Groq, shows the answer
```

Your vector database persists across restarts in a Docker named volume — you do not need to re-ingest documents every time you start the app.

---

## Supported file types

Docling handles: PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), HTML, and plain text.

---

## Troubleshooting

**"Docker is not running"**
Open Docker Desktop and wait for it to finish starting, then try again.

**The browser opens but shows a connection error**
The frontend container is still booting. Wait 10–15 seconds and refresh.

**Answers seem wrong or off-topic**
Make sure you ran `add-documents.command` after adding your files. Re-running it is safe — existing documents are updated, not duplicated.

**"No documents found"**
Check that your files are directly inside the `documents/` folder (not in a subfolder).
