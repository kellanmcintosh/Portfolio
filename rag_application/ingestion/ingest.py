import os
import sys
from pathlib import Path

import requests
import tiktoken
import chromadb
from docling.document_converter import DocumentConverter

DOCUMENTS_DIR = Path("/documents")
COLLECTION_NAME = "documents"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50
EMBED_MODEL = "text-embedding-004"

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1"
tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_TOKENS
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    url = f"{GEMINI_BASE}/models/{EMBED_MODEL}:batchEmbedContents"
    body = {
        "requests": [
            {
                "model": f"models/{EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": "RETRIEVAL_DOCUMENT",
            }
            for t in texts
        ]
    }
    response = requests.post(url, params={"key": GEMINI_API_KEY}, json=body)
    response.raise_for_status()
    return [e["values"] for e in response.json()["embeddings"]]


def main():
    files = [f for f in DOCUMENTS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        print("No documents found in /documents — nothing to ingest.")
        sys.exit(0)

    print(f"Found {len(files)} file(s) to ingest.")

    converter = DocumentConverter()
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    for file_path in files:
        print(f"  Converting: {file_path.name}")
        try:
            result = converter.convert(str(file_path))
            markdown = result.document.export_to_markdown()
        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")
            continue

        chunks = chunk_text(markdown)
        print(f"  Chunked into {len(chunks)} chunk(s)")

        ids = [f"{file_path.name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]
        embeddings = embed(chunks)

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  Upserted {len(chunks)} chunk(s) from {file_path.name}")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
