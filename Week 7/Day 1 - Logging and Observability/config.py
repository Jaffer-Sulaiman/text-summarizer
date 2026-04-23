"""
config.py — Centralized Configuration Layer  (Week 7 Day 1 — Logging & Observability)
=======================================================================================
Extends Day 2 config with observability constants:
  - LOG_LEVEL        : controls verbosity for stdout + file
  - LOG_DIR          : directory where rotating log files are written
  - LOG_FILE_MAX_BYTES  : rotate when file reaches this size (default 10 MB)
  - LOG_FILE_BACKUP_COUNT : number of archived log files to keep (default 5)
  - LOG_PROMPT       : whether to write full prompt text to log (default False — privacy)
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
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "logistics_support_kb_v3")

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
# Failure Handling Constants  (from Day 2 — unchanged)
# ===========================================================================

LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_DELAY: float = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
GRAPH_TIMEOUT_SECONDS: float = float(os.getenv("GRAPH_TIMEOUT_SECONDS", "90"))
VECTORSTORE_TIMEOUT_SECONDS: float = float(os.getenv("VECTORSTORE_TIMEOUT_SECONDS", "10"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = int(
    os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")
)
CIRCUIT_BREAKER_RESET_TIMEOUT: float = float(
    os.getenv("CIRCUIT_BREAKER_RESET_TIMEOUT", "60")
)

# ===========================================================================
# ★ NEW IN WEEK 7 DAY 1 — Observability Constants
# ===========================================================================

# Minimum log level emitted to both stdout and the rotating log file.
# Values: "DEBUG" | "INFO" | "WARNING" | "ERROR"
# Set to "DEBUG" locally for full prompt + timing detail; keep "INFO" in production.
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Directory where rotating log files are stored.
# Default: <project_root>/logs/
LOG_DIR: str = os.getenv(
    "LOG_DIR",
    os.path.join(_base_dir, "logs"),
)

# Rotate the log file when it reaches this size (bytes). Default = 10 MB.
LOG_FILE_MAX_BYTES: int = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))

# Number of rotated backup log files to retain (rag_agent.log.1 … .N).
LOG_FILE_BACKUP_COUNT: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

# If "true", full prompt text is written to the log file at DEBUG level.
# Disable in production to avoid logging customer PII / query data.
# Controlled by observability.py — this constant documents the env variable name.
LOG_PROMPT: bool = os.getenv("LOG_PROMPT", "false").lower() == "true"
