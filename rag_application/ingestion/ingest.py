"""One-shot ingestion job: documents/ → markdown → chunks → embeddings → ChromaDB.

Run via `scripts/add-documents.command`. Idempotent: re-running with the same
files updates rather than duplicates them (chunk IDs are derived from filename
+ index, and the collection is upserted, not appended).
"""

import logging
import os
import sys
from pathlib import Path

import chromadb
import tiktoken
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("/documents")
COLLECTION_NAME = "documents"
CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50

CHROMA_HOST = os.environ["CHROMA_HOST"]
CHROMA_PORT = int(os.environ["CHROMA_PORT"])

tokenizer = tiktoken.get_encoding("cl100k_base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str) -> list[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_TOKENS
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    logger.debug("Produced %d chunk(s) from %d token(s)", len(chunks), len(tokens))
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    logger.debug("Embedding %d text(s)", len(texts))
    return embed_model.encode(texts, normalize_embeddings=True).tolist()


def main():
    files = [f for f in DOCUMENTS_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        logger.warning("No documents found in %s — nothing to ingest.", DOCUMENTS_DIR)
        sys.exit(0)

    logger.info("Found %d file(s) to ingest", len(files))

    converter = DocumentConverter()
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    for file_path in files:
        logger.info("Converting: %s", file_path.name)
        try:
            result = converter.convert(str(file_path))
            markdown = result.document.export_to_markdown()
        except Exception:
            logger.exception("Skipping %s — conversion failed", file_path.name)
            continue

        chunks = chunk_text(markdown)
        if not chunks:
            logger.warning("No chunks produced from %s — skipping", file_path.name)
            continue

        logger.info("Chunked %s into %d chunk(s)", file_path.name, len(chunks))

        ids = [f"{file_path.name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]
        embeddings = embed(chunks)

        collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.info("Upserted %d chunk(s) from %s", len(chunks), file_path.name)

    logger.info("Ingestion complete. Total chunks in collection: %d", collection.count())


if __name__ == "__main__":
    main()
