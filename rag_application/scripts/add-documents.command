#!/bin/bash

cd "$(dirname "$0")/.."

echo ""
echo "========================================="
echo "  RAG Application — Ingest Documents"
echo "========================================="
echo ""

if ! docker info > /dev/null 2>&1; then
    echo "❌  Docker is not running."
    echo "    Please open Docker Desktop and try again."
    echo ""
    exit 1
fi

doc_count=$(find documents -maxdepth 1 -type f ! -name ".*" | wc -l | tr -d ' ')

if [ "$doc_count" -eq 0 ]; then
    echo "⚠️   No documents found in the documents/ folder."
    echo "    Add PDF, Word, or text files there and run this script again."
    echo ""
    exit 0
fi

echo "📄  Found $doc_count document(s) — building and starting ingestion..."
echo ""

docker compose run --rm --build ingestion 2>&1 | grep -v "^$" | sed 's/^/    /'

echo ""
echo "✅  Ingestion complete. You can now ask questions in the chat."
echo ""
