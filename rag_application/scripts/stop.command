#!/bin/bash

cd "$(dirname "$0")/.."

echo ""
echo "========================================="
echo "  RAG Application — Stop"
echo "========================================="
echo ""

if ! docker info > /dev/null 2>&1; then
    echo "ℹ️   Docker is not running — nothing to stop."
    echo ""
    exit 0
fi

echo "⏳  Stopping all services..."

docker compose down > /dev/null 2>&1

echo "✅  All services stopped."
echo "    Your ChromaDB data is preserved in the chroma_data volume."
echo ""
