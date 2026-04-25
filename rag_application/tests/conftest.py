import sys
import os
from pathlib import Path

# Required before any service module is imported
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8000")

# Make ingestion/ and frontend/ importable
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "ingestion"))

# Mock streamlit before frontend/app.py is imported anywhere
from unittest.mock import MagicMock
sys.modules["streamlit"] = MagicMock()
sys.path.insert(0, str(root / "frontend"))
