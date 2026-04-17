"""
config.py — Centralized Configuration Layer  (Day 2 — Failure Handling)
========================================================================
Extends Day 1 config with failure-handling constants:
  - LLM call timeout
  - Retry parameters
  - Circuit breaker thresholds
  - Full pipeline timeout (graph-level)
"""

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Required
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "[CONFIG] GOOGLE_API_KEY is not set. "
        "Add it to your .env file before starting the server."
    )

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
_base_dir = os.path.dirname(__file__)
CHROMA_DB_DIR: str = os.getenv(
    "CHROMA_DB_DIR",
    os.path.join(_base_dir, ".chroma_db"),
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "logistics_support_kb_v2")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_KEY: str = os.getenv("API_KEY", "")

# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_RPM: int = int(os.getenv("RATE_LIMIT_RPM", "60"))

# ---------------------------------------------------------------------------
# File Ingestion
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: tuple = (".pdf", ".txt", ".docx")

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH: int = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
MIN_QUERY_LENGTH: int = 1

# ---------------------------------------------------------------------------
# Session Memory
# ---------------------------------------------------------------------------
SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "7200"))
SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "20"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_SIMPLE: int = 3
TOP_K_COMPLEX: int = 6

# ---------------------------------------------------------------------------
# Domain
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
# UI
# ---------------------------------------------------------------------------
UI_PORT: int = int(os.getenv("UI_PORT", "7860"))
UI_API_BASE: str = os.getenv("UI_API_BASE", f"http://localhost:{API_PORT}")

# ===========================================================================
# ★ NEW IN DAY 2 — Failure Handling Constants
# ===========================================================================

# LLM timeout: maximum seconds to wait for a single LLM call before raising
# LLMTimeoutError.  Set conservatively high for structured output calls.
LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

# Retry: how many times to retry a transient-failing LLM call
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Retry: initial back-off delay in seconds (doubles each attempt)
LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))

# Full pipeline timeout: maximum seconds for the complete LangGraph invocation
# before the API layer returns HTTP 504 Gateway Timeout.
GRAPH_TIMEOUT_SECONDS: float = float(os.getenv("GRAPH_TIMEOUT_SECONDS", "90"))

# ChromaDB retrieval timeout in seconds (wraps the blocking Chroma call)
VECTORSTORE_TIMEOUT_SECONDS: float = float(os.getenv("VECTORSTORE_TIMEOUT_SECONDS", "10"))

# Circuit breaker: how many consecutive failures before opening the circuit
CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(
    os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
)

# Circuit breaker: seconds to wait in OPEN state before probing again
CIRCUIT_BREAKER_RESET_TIMEOUT: float = float(
    os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", "60")
)
