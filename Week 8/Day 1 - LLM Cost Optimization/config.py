"""
config.py — Centralized Configuration Layer  (Week 8 Day 1 — LLM Cost Optimization)
======================================================================================
Extends Week 7 Day 1 config with cost-optimization constants:

  NEW in Week 8 Day 1:
  - LLM_CLASSIFIER_MODEL  : cheaper/faster model for all classification nodes
  - LLM_ANSWER_MODEL      : full-quality model for answer generation only
  - MAX_CONTEXT_TOKENS    : hard cap on tokens fed to grade_relevance + generate_answer
  - MAX_TOKENS_PER_REQUEST: budget warning threshold per single request
  - MAX_HISTORY_REPHRASE  : history messages passed to rephrase_query (tightened 6→4)
  - MAX_HISTORY_ANSWER    : history messages passed to generate_answer (tightened 4→2)

  Inherited from Week 7 Day 1:
  - LOG_LEVEL / LOG_DIR / LOG_FILE_* / LOG_PROMPT (observability)
  - LLM_TIMEOUT / LLM_MAX_RETRIES / CIRCUIT_BREAKER_* (resilience)
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
# LLM — Base (Week 7 compatible)
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ===========================================================================
# ★ NEW IN WEEK 8 DAY 1 — Model Tiering
# ===========================================================================
# Cheaper, faster model for pure classification tasks (intent, complexity, grade).
# gemini-2.0-flash is ~4× cheaper per token than the full model.
LLM_CLASSIFIER_MODEL: str = os.getenv("LLM_CLASSIFIER_MODEL", "gemini-2.0-flash")

# Full-quality model reserved only for answer generation.
LLM_ANSWER_MODEL: str = os.getenv("LLM_ANSWER_MODEL", "gemini-2.0-flash")

# ===========================================================================
# ★ NEW IN WEEK 8 DAY 1 — Context Budget
# ===========================================================================
# Hard cap on tokens fed to grade_relevance and generate_answer.
# Chunks are truncated to fit within this limit before building the prompt.
# Estimated: 2000 tokens ≈ 8000 characters of retrieved context.
MAX_CONTEXT_TOKENS: int = int(os.getenv("MAX_CONTEXT_TOKENS", "2000"))

# ===========================================================================
# ★ NEW IN WEEK 8 DAY 1 — Token Budget per Request
# ===========================================================================
# If total tokens_in + tokens_out for a single request exceeds this threshold,
# a WARNING is emitted in the logs. Does not block the request.
MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "4000"))

# ===========================================================================
# ★ NEW IN WEEK 8 DAY 1 — History Window Tightening
# ===========================================================================
# Number of history messages (not turns) passed to rephrase_query.
# Was 6 in Week 7. Reduced to 4 to save tokens.
MAX_HISTORY_REPHRASE: int = int(os.getenv("MAX_HISTORY_REPHRASE", "4"))

# Number of history messages (not turns) passed to generate_answer.
# Was 4 in Week 7. Reduced to 2 to save tokens.
MAX_HISTORY_ANSWER: int = int(os.getenv("MAX_HISTORY_ANSWER", "2"))

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
# Observability Constants  (from Week 7 Day 1 — unchanged)
# ===========================================================================

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR: str = os.getenv(
    "LOG_DIR",
    os.path.join(_base_dir, "logs"),
)
LOG_FILE_MAX_BYTES: int = int(os.getenv("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_FILE_BACKUP_COUNT: int = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
LOG_PROMPT: bool = os.getenv("LOG_PROMPT", "false").lower() == "true"
