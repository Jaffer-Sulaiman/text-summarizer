"""
logger.py — Structured JSON Logging Layer
==========================================
Provides a centralized, structured logger that emits JSON-formatted log lines.
Each entry includes timestamp, level, component name, and any extra fields passed
by the caller (session_id, latency_ms, node_name, etc.).

Usage:
    from logger import get_logger
    log = get_logger("vectorstore")
    log.info("Document ingested", extra={"source": "policy.pdf", "chunks": 42})
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Formats each LogRecord as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }

        # Merge any extra key-value pairs the caller passed in
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                if key not in ("msg", "args", "levelname", "name", "pathname",
                               "filename", "module", "exc_info", "exc_text",
                               "stack_info", "lineno", "funcName", "created",
                               "msecs", "relativeCreated", "thread",
                               "threadName", "processName", "process",
                               "message", "taskName"):
                    payload[key] = val

        # Append exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger configured with the JSON formatter.
    Calling this multiple times with the same name returns the same instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


class TimingContext:
    """
    Context manager that measures elapsed wall-clock time and logs a
    structured completion entry.

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
