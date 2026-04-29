"""
observability.py — Production-Grade Logging & Observability Engine
==================================================================
Week 7 Day 1 — new module.

Provides the observability layer for the entire RAG pipeline:
  1. Log initialisation  — stdout + RotatingFileHandler wired once at startup
  2. Trace ID            — UUID4 per request, threaded through all nodes
  3. Prompt logging      — controlled by LOG_PROMPT env var (privacy-safe default: off)
  4. Token estimation    — lightweight ~4 chars/token heuristic (no external dep)
  5. Structured audit    — log_llm_call(), log_retrieval() emit standardised records
  6. TracingContext       — enhanced TimingContext that carries trace_id

Usage:
    from observability import (
        init_logging, new_trace_id, log_llm_call,
        log_retrieval, log_prompt, estimate_tokens, TracingContext,
    )
"""

import hashlib
import logging
import os
import traceback as tb
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal sentinel so init runs only once
# ---------------------------------------------------------------------------
_INITIALISED = False


# ===========================================================================
# 1. LOG INITIALISATION
# ===========================================================================

def init_logging(
    log_dir: str,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Wire a RotatingFileHandler to the root logger so that ALL named loggers
    (api, graph, vectorstore …) automatically write to disk as well as stdout.

    Call this exactly once at application startup (api.py __main__ or top-level).
    Subsequent calls are no-ops.
    """
    global _INITIALISED
    if _INITIALISED:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "rag_agent.log")

    # Import here to avoid circular — logger.py imports observability constants
    from logger import _JsonFormatter  # noqa: PLC0415

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.addHandler(file_handler)

    numeric = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(numeric)

    _INITIALISED = True

    # Log the init event on the root logger
    logging.getLogger("observability").info(
        "Observability initialised",
        extra={
            "log_file": log_path,
            "log_level": log_level,
            "max_bytes": max_bytes,
            "backup_count": backup_count,
        },
    )


# ===========================================================================
# 2. TRACE ID
# ===========================================================================

def new_trace_id() -> str:
    """Generate a UUID4 trace ID for one end-to-end request."""
    return str(uuid.uuid4())


# ===========================================================================
# 3. TOKEN ESTIMATION
# ===========================================================================

# Industry rule of thumb: 1 token ≈ 4 characters for English text.
# Gemini tokeniser is slightly different but the 4-char heuristic is accurate
# enough for observability purposes without adding a tokeniser dependency.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Return an estimated token count for *text* using the 4-char heuristic."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ===========================================================================
# 4. PROMPT LOGGING
# ===========================================================================

# Environment flag — set LOG_PROMPT=true to write prompts to disk.
# Defaults to False so customer data / PII does not land in log files in prod.
_LOG_PROMPT_ENABLED: Optional[bool] = None

_PROMPT_PREVIEW_CHARS = 300   # characters to include in the INFO preview


def _prompt_logging_enabled() -> bool:
    global _LOG_PROMPT_ENABLED
    if _LOG_PROMPT_ENABLED is None:
        _LOG_PROMPT_ENABLED = os.getenv("LOG_PROMPT", "false").lower() == "true"
    return _LOG_PROMPT_ENABLED


def _prompt_hash(text: str) -> str:
    """SHA-256 prefix for prompt de-duplication / change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_prompt(
    log: logging.Logger,
    *,
    trace_id: str,
    node: str,
    prompt: str,
    max_chars: int = _PROMPT_PREVIEW_CHARS,
) -> None:
    """
    Log prompt metadata at INFO and (if LOG_PROMPT=true) the full text at DEBUG.

    Always logged:
      - prompt_chars   : total character count
      - tokens_estimate: rough token count
      - prompt_hash    : first 16 hex chars of SHA-256 (deduplication aid)
      - prompt_preview : first max_chars characters (safe for INFO)

    Only when LOG_PROMPT=true appended at DEBUG:
      - prompt_full    : the complete prompt string
    """
    tokens = estimate_tokens(prompt)
    preview = prompt[:max_chars].replace("\n", " ↵ ")
    extra: Dict[str, Any] = {
        "trace_id": trace_id,
        "node": node,
        "prompt_chars": len(prompt),
        "tokens_estimate": tokens,
        "prompt_hash": _prompt_hash(prompt),
        "prompt_preview": preview,
    }
    log.info("Prompt prepared", extra=extra)

    if _prompt_logging_enabled():
        log.debug("Prompt full text", extra={**extra, "prompt_full": prompt})


# ===========================================================================
# 5. LLM CALL AUDIT
# ===========================================================================

def log_llm_call(
    log: logging.Logger,
    *,
    trace_id: str,
    node: str,
    model: str,
    tokens_in: int,
    latency_ms: float,
    success: bool,
    tokens_out: int = 0,
    attempt: int = 1,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    traceback_str: Optional[str] = None,
) -> None:
    """
    Emit a standardised LLM audit record.

    Fields logged:
      - trace_id, node, model
      - tokens_in, tokens_out, total_tokens
      - latency_ms, success
      - attempt (retry attempt number)
      - error, error_type, traceback (on failure)
    """
    extra: Dict[str, Any] = {
        "trace_id": trace_id,
        "node": node,
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "latency_ms": latency_ms,
        "success": success,
        "attempt": attempt,
    }
    if not success:
        extra["error"] = error or "unknown"
        extra["error_type"] = error_type or "unknown"
        if traceback_str:
            extra["traceback"] = traceback_str
        log.error("LLM call failed", extra=extra)
    else:
        log.info("LLM call completed", extra=extra)


# ===========================================================================
# 6. RETRIEVAL AUDIT
# ===========================================================================

def log_retrieval(
    log: logging.Logger,
    *,
    trace_id: str,
    query: str,
    top_k: int,
    category_filter: Optional[str],
    chunks_returned: int,
    latency_ms: float,
    chunk_sizes: Optional[List[int]] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """
    Emit a standardised retrieval audit record so operators can track:
      - what query was issued, with what filters
      - how many chunks came back and their sizes
      - how long the ChromaDB call took
    """
    extra: Dict[str, Any] = {
        "trace_id": trace_id,
        "query_chars": len(query),
        "query_preview": query[:150].replace("\n", " ↵ "),
        "top_k_requested": top_k,
        "category_filter": category_filter or "none",
        "chunks_returned": chunks_returned,
        "latency_ms": latency_ms,
        "success": success,
    }
    if chunk_sizes:
        extra["chunk_sizes"] = chunk_sizes
        extra["avg_chunk_chars"] = round(sum(chunk_sizes) / len(chunk_sizes))
        extra["total_context_chars"] = sum(chunk_sizes)
        extra["total_context_tokens"] = estimate_tokens(" " * extra["total_context_chars"])
    if not success and error:
        extra["error"] = error
    log.info("Retrieval completed", extra=extra)


# ===========================================================================
# 7. TRACING CONTEXT (enhanced TimingContext)
# ===========================================================================

class TracingContext:
    """
    Context manager that:
      1. Records wall-clock time for an operation
      2. Emits a structured log entry on exit with latency_ms + trace_id
      3. Captures any exception traceback as a structured field

    Usage:
        with TracingContext(log, "retrieve", trace_id=tid, session_id=sid):
            docs = vectorstore.retrieve(query)
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        trace_id: str = "",
        **extra_fields,
    ):
        self._log = logger
        self._op = operation
        self._trace_id = trace_id
        self._extra = extra_fields
        self._start: float = 0.0

    def __enter__(self):
        import time
        self._start = time.perf_counter()
        self._log.debug(
            f"{self._op} started",
            extra={"operation": self._op, "trace_id": self._trace_id, **self._extra},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        base = {
            "operation": self._op,
            "trace_id": self._trace_id,
            "latency_ms": elapsed_ms,
            **self._extra,
        }
        if exc_type is None:
            self._log.info(f"{self._op} completed", extra=base)
        else:
            self._log.error(
                f"{self._op} failed",
                extra={
                    **base,
                    "error": str(exc_val),
                    "error_type": exc_type.__name__,
                    "traceback": tb.format_exc(),
                },
            )
        return False  # Do not suppress exceptions


# ===========================================================================
# 8. TAIL LOG LINES  (used by /logs API endpoint)
# ===========================================================================

def tail_log_file(log_dir: str, last_n: int = 100) -> List[str]:
    """
    Return the last *last_n* lines from the active log file.
    Returns an empty list if the log file does not yet exist.
    """
    log_path = os.path.join(log_dir, "rag_agent.log")
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip("\n") for l in lines[-last_n:]]
    except Exception as exc:
        return [f"[tail_log_file error] {exc}"]
