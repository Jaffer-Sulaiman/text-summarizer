"""
logger.py — Structured JSON Logging Layer  (Week 7 Day 1 — Logging & Observability)
======================================================================================
Extends Week 6 Day 2 with:
  - LOG_LEVEL env var respected on every logger
  - RotatingFileHandler wired via observability.init_logging() (called once at startup)
  - trace_id passed transparently through _JsonFormatter
  - TracingContext re-exported for backward-compatibility with graph.py callers

The _JsonFormatter class is imported by observability.py to reuse the same
JSON serialisation for the file handler without duplication.

Usage (unchanged from Day 1/2):
    from logger import get_logger, TimingContext
    log = get_logger("vectorstore")
    log.info("Document ingested", extra={"source": "policy.pdf", "chunks": 42})

New usage (trace_id):
    log.info("Node started", extra={"trace_id": trace_id, "session_id": sid})
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# JSON Formatter  (also used by observability.py file handler)
# ---------------------------------------------------------------------------
class _JsonFormatter(logging.Formatter):
    """Formats each LogRecord as a single JSON object on one line."""

    # Fields that come from LogRecord internals — we never want to re-emit them
    _SKIP = frozenset({
        "msg", "args", "levelname", "name", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno",
        "funcName", "created", "msecs", "relativeCreated", "thread",
        "threadName", "processName", "process", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }

        # Merge any extra key-value pairs the caller passed in.
        # trace_id is treated specially — it goes right after "msg" for readability.
        trace_id = None
        extras: dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key in self._SKIP or key.startswith("_"):
                continue
            if key == "trace_id":
                trace_id = val
            else:
                extras[key] = val

        if trace_id is not None:
            payload["trace_id"] = trace_id
        payload.update(extras)

        # Append exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Return a named logger configured with the JSON formatter on stdout.

    The file handler is attached separately by observability.init_logging()
    which is called once at startup. Calling get_logger() multiple times with
    the same name is safe — the same instance is returned.

    Args:
        name:  Logger name (used as ``component`` field in JSON output).
        level: Optional explicit level. If None, reads LOG_LEVEL env var
               (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False   # Don't double-emit to root logger's stdout

    if level is not None:
        logger.setLevel(level)
    else:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        numeric = getattr(logging, env_level, logging.INFO)
        logger.setLevel(numeric)

    return logger


# ---------------------------------------------------------------------------
# TimingContext  (unchanged API — kept for backward-compat with existing callers)
# ---------------------------------------------------------------------------
class TimingContext:
    """
    Context manager that measures elapsed wall-clock time and logs a
    structured completion entry.

    For new code prefer TracingContext from observability.py which also
    carries trace_id and captures tracebacks on failure.

    Usage:
        with TimingContext(log, "retrieve", session_id="abc123"):
            docs = vectorstore.retrieve(query)
    """

    def __init__(self, logger: logging.Logger, operation: str, **extra_fields):
        self._log = logger
        self._op = operation
        self._extra = extra_fields
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        if exc_type is None:
            self._log.info(
                f"{self._op} completed",
                extra={"operation": self._op, "latency_ms": elapsed_ms, **self._extra},
            )
        else:
            self._log.error(
                f"{self._op} failed",
                extra={
                    "operation": self._op,
                    "latency_ms": elapsed_ms,
                    "error": str(exc_val),
                    **self._extra,
                },
            )
        return False  # Let exceptions propagate
