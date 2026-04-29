"""
config.py — Centralized Configuration Layer  (Week 8 Day 2 — AI Knowledge Assistant)
======================================================================================
General-purpose document QA system — no domain-specific taxonomy.

Inherits all resilience + logging + cost-optimization constants from Week 8 Day 1.

NEW in Week 8 Day 2:
  - MAX_FILES_PER_BULK       : cap on files accepted in a single /ingest/bulk call
  - ALLOWED_EXTENSIONS       : .pdf and .txt only (clean, minimal surface area)
  - Removed DOCUMENT_CATEGORIES taxonomy — this assistant is domain-agnostic
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
# LLM — Model Tiering  (inherited from Week 8 Day 1)
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Cheaper/faster model for all classification nodes (intent, grade).
LLM_CLASSIFIER_MODEL: str = os.getenv("LLM_CLASSIFIER_MODEL", "gemini-2.0-flash")

# Full-quality model reserved only for answer generation.
LLM_ANSWER_MODEL: str = os.getenv("LLM_ANSWER_MODEL", "gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Context Budget  (inherited from Week 8 Day 1)
# ---------------------------------------------------------------------------
# Hard cap on tokens fed to grade_relevance and generate_answer.
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))

# Per-request budget warning threshold.
MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000"))

# ---------------------------------------------------------------------------
# History Window  (inherited from Week 8 Day 1)
# ---------------------------------------------------------------------------
MAX_HISTORY_REPHRASE: int = int(os.getenv("MAX_HISTORY_REPHRASE", "4"))
MAX_HISTORY_ANSWER: int = int(os.getenv("MAX_HISTORY_ANSWER", "2"))

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
_base_dir = os.path.dirname(__file__)
CHROMA_DB_DIR: str = os.getenv(
    "CHROMA_DB_DIR",
    os.path.join(_base_dir, ".chroma_db"),
)
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "knowledge_assistant_v1")

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

# ★ NEW Week 8 Day 2: max files in a single bulk upload request
MAX_FILES_PER_BULK: int = int(os.getenv("MAX_FILES_PER_BULK", "10"))

# ★ Week 8 Day 2: general-purpose (.pdf, .txt only)
ALLOWED_EXTENSIONS: tuple = (".pdf", ".txt")

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
# UI
# ---------------------------------------------------------------------------
UI_PORT: int = int(os.getenv("UI_PORT", "7860"))
UI_API_BASE: str = os.getenv("UI_API_BASE", f"http://localhost:{API_PORT}")

# ---------------------------------------------------------------------------
# Resilience  (inherited from Week 8 Day 1)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Observability  (inherited from Week 8 Day 1)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR: str = os.getenv(
    "LOG_DIR",
    os.path.join(_base_dir, "logs"),
)
LOG_FILE_MAX_BYTES: int = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_FILE_BACKUP_COUNT: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
LOG_PROMPT: bool = os.getenv("LOG_PROMPT", "false").lower() == "true"
