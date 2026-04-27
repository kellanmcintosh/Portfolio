import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Required before any service module is imported
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8000")

# Mock heavy deps so tests run without installing them
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["streamlit"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["docling"] = MagicMock()
sys.modules["docling.document_converter"] = MagicMock()

# Make ingestion/ and frontend/ importable
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "ingestion"))
sys.path.insert(0, str(root / "frontend"))
