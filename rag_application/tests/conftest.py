import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# Required before any service module is imported
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8000")

# Mock heavy deps so tests run without installing them
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["streamlit"] = MagicMock()

# Make ingestion/ and frontend/ importable
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "ingestion"))
sys.path.insert(0, str(root / "frontend"))
