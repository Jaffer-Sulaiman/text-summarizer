"""
config.py — Centralized Configuration Layer
============================================
Single source of truth for all environment variables and application constants.
Fails fast at startup if required variables are missing.
"""

import os
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------------
# Load .env (walks up directory tree to find it)
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Required — will raise clear error if missing
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "[CONFIG] GOOGLE_API_KEY is not set. "
        "Add it to your .env file before starting the server."
    )

# ---------------------------------------------------------------------------
# LLM Settings
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# ChromaDB Settings
# ---------------------------------------------------------------------------
_base_dir = os.path.dirname(__file__)
CHROMA_DB_DIR: str = os.getenv(
    "CHROMA_DB_DIR",
    os.path.join(_base_dir, ".chroma_db"),
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "logistics_support_kb")

# ---------------------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
# Optional static API key for client auth (leave blank to disable auth)
API_KEY: str = os.getenv("API_KEY", "")

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "60"))   # requests per minute per IP

# ---------------------------------------------------------------------------
# File Ingestion Constraints
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: tuple = (".pdf", ".txt", ".docx")

# ---------------------------------------------------------------------------
# Query Constraints
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
MIN_QUERY_LENGTH: int = 1

# ---------------------------------------------------------------------------
# Session Memory
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "7200"))   # 2 hours
SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "20"))          # sliding window

# ---------------------------------------------------------------------------
# Retrieval Settings
# ---------------------------------------------------------------------------
TOP_K_SIMPLE: int = 3
TOP_K_COMPLEX: int = 6

# ---------------------------------------------------------------------------
# Document Categories (logistics domain)
# ---------------------------------------------------------------------------
DOCUMENT_CATEGORIES: list = [
    "shipping_policy",
    "tracking",
    "rates_and_zones",
    "customs_and_compliance",
    "claims_and_disputes",
    "faq",
    "general",
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
UI_PORT: int = int(os.getenv("UI_PORT", "7860"))
UI_API_BASE: str = os.getenv("UI_API_BASE", f"http://localhost:{API_PORT}")
