#!/bin/bash

cd "$(dirname "$0")/.."

echo ""
echo "========================================="
echo "  RAG Application — Start"
echo "========================================="
echo ""

if ! docker info > /dev/null 2>&1; then
    echo "❌  Docker is not running."
    echo "    Please open Docker Desktop and try again."
    echo ""
    exit 1
fi

echo "✅  Docker is running"
echo "⏳  Building and starting ChromaDB and frontend..."

if ! docker compose up -d --build chromadb frontend > /dev/null 2>&1; then
    echo "❌  Failed to start services. Check that .env exists and Docker has enough resources."
    echo ""
    exit 1
fi

echo "✅  Services started — waiting for frontend to be ready..."

ready=false
for i in {1..20}; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 2
done

if [ "$ready" = false ]; then
    echo "⚠️   Frontend is taking longer than expected."
    echo "    Try opening http://localhost:8501 manually in a moment."
else
    echo "✅  Frontend is ready"
fi

echo ""
echo "🌐  Opening http://localhost:8501"
echo ""
open http://localhost:8501
