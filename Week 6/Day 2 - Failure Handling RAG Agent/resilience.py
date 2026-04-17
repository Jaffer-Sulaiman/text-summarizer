"""
resilience.py — Failure Handling & Resilience Primitives
==========================================================
Central module providing battle-tested resilience patterns for every
external I/O boundary in the system.

Patterns implemented:
  1. Timeout wrapper      — prevents LLM calls from hanging indefinitely
  2. Retry w/ backoff     — handles transient network / quota errors
  3. Circuit Breaker      — stops cascading failures when a service is down

Exception hierarchy (raised by this module and caught by graph nodes):
  ResilienceError
    ├── LLMTimeoutError       — call exceeded LLM_TIMEOUT_SECONDS
    ├── LLMRateLimitError     — HTTP 429 / quota exceeded
    ├── LLMAuthError          — invalid / expired API key (401 / 403)
    ├── LLMUnavailableError   — network error reaching Google API
    ├── RetryExhaustedError   — all retry attempts failed
    ├── CircuitOpenError      — circuit breaker is OPEN, call rejected
    └── VectorStoreError      — ChromaDB read / write / connect failure
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum
from typing import Callable, Optional, TypeVar

from logger import get_logger

log = get_logger("resilience")
T = TypeVar("T")


# ==============================================================================
# EXCEPTION HIERARCHY
# ==============================================================================

class ResilienceError(Exception):
    """Base for all resilience-layer errors. Carries a machine-readable code."""
    def __init__(self, message: str, code: str = "RESILIENCE_ERROR"):
        super().__init__(message)
        self.code = code


class LLMTimeoutError(ResilienceError):
    def __init__(self, message: str):
        super().__init__(message, code="LLM_TIMEOUT")


class LLMRateLimitError(ResilienceError):
    def __init__(self, message: str):
        super().__init__(message, code="LLM_RATE_LIMITED")


class LLMAuthError(ResilienceError):
    def __init__(self, message: str):
        super().__init__(message, code="LLM_AUTH_ERROR")


class LLMUnavailableError(ResilienceError):
    def __init__(self, message: str):
        super().__init__(message, code="LLM_UNAVAILABLE")


class RetryExhaustedError(ResilienceError):
    def __init__(self, message: str, attempts: int):
        super().__init__(message, code="RETRY_EXHAUSTED")
        self.attempts = attempts


class CircuitOpenError(ResilienceError):
    def __init__(self, service: str, reset_in: float):
        super().__init__(
            f"Circuit breaker for '{service}' is OPEN. Retry in {reset_in:.0f}s.",
            code="CIRCUIT_OPEN",
        )
        self.service = service
        self.reset_in = reset_in


class VectorStoreError(ResilienceError):
    def __init__(self, message: str):
        super().__init__(message, code="VECTOR_STORE_ERROR")


# Errors that should NEVER be retried (retrying won't help)
_NON_RETRYABLE = (LLMTimeoutError, LLMAuthError, CircuitOpenError)

# Errors that MAY succeed on retry (transient)
_RETRYABLE = (LLMRateLimitError, LLMUnavailableError)


# ==============================================================================
# ERROR CLASSIFIER
# ==============================================================================

def classify_llm_error(exc: Exception) -> ResilienceError:
    """
    Map any raw exception from LangChain / Google GenAI SDK into our typed
    hierarchy by inspecting the error message string.
    This is intentionally broad to handle different SDK versions.
    """
    if isinstance(exc, ResilienceError):
        return exc  # Already typed

    msg = str(exc).lower()
    raw = str(exc)

    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return LLMTimeoutError(f"LLM timed out: {raw}")

    if "429" in msg or "quota" in msg or ("rate" in msg and "limit" in msg):
        return LLMRateLimitError(f"LLM quota/rate-limit: {raw}")

    if any(k in msg for k in ("401", "403", "api key", "api_key",
                               "authentication", "invalid key", "credentials")):
        return LLMAuthError(f"LLM auth failure: {raw}")

    if any(k in msg for k in ("context", "token")) and any(
        k in msg for k in ("length", "window", "limit", "overflow", "too long")
    ):
        # Context window exceeded — treat as unavailable (no retry worth attempting)
        return LLMUnavailableError(f"Context window overflow: {raw}")

    if any(k in msg for k in ("connection", "network", "unreachable",
                               "connection refused", "eof", "broken pipe",
                               "reset by peer", "socket")):
        return LLMUnavailableError(f"Network error: {raw}")

    # Generic unknown error — try once more
    return LLMUnavailableError(f"LLM call failed: {raw}")


# ==============================================================================
# 1. TIMEOUT WRAPPER
# ==============================================================================

def invoke_with_timeout(
    fn: Callable[..., T],
    *args,
    timeout_seconds: float = 30.0,
    **kwargs,
) -> T:
    """
    Execute fn(*args, **kwargs) in a thread and raise LLMTimeoutError if it
    does not complete within timeout_seconds.

    Note: The underlying thread cannot be forcibly cancelled in Python — it
    will continue running in the background after the timeout. This is
    acceptable because LangChain's network calls will eventually resolve.
    In production, set appropriate HTTP client timeouts on the Google SDK.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            log.warning(
                "LLM call timed out",
                extra={"timeout_seconds": timeout_seconds, "fn": getattr(fn, "__name__", str(fn))},
            )
            raise LLMTimeoutError(
                f"LLM did not respond within {timeout_seconds}s"
            )
        except Exception as exc:
            raise exc  # Re-raise for caller to classify


# ==============================================================================
# 2. RETRY WITH EXPONENTIAL BACKOFF
# ==============================================================================

def invoke_with_retry(
    fn: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout_seconds: float = 30.0,
    **kwargs,
) -> T:
    """
    Call fn with a timeout, retrying on transient failures with exponential
    back-off. Non-retryable errors (timeout, auth) surface immediately.

    Back-off schedule (base_delay=1.0): 1s → 2s → 4s → give up
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            result = invoke_with_timeout(fn, *args,
                                         timeout_seconds=timeout_seconds,
                                         **kwargs)
            if attempt > 1:
                log.info(
                    "LLM call succeeded after retry",
                    extra={"attempt": attempt},
                )
            return result

        except _NON_RETRYABLE as exc:
            log.error(
                "Non-retryable LLM error — aborting",
                extra={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise  # Immediately surface without retry

        except Exception as exc:
            typed = classify_llm_error(exc)

            if isinstance(typed, _NON_RETRYABLE):
                raise typed

            last_error = typed

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
                log.warning(
                    "LLM call failed — retrying",
                    extra={
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "retry_in_seconds": delay,
                        "error": str(typed),
                    },
                )
                time.sleep(delay)
            else:
                log.error(
                    "All LLM retries exhausted",
                    extra={"attempts": max_retries, "last_error": str(typed)},
                )

    raise RetryExhaustedError(
        f"LLM call failed after {max_retries} attempts. Last: {last_error}",
        attempts=max_retries,
    )


# ==============================================================================
# 3. CIRCUIT BREAKER
# ==============================================================================

class _CircuitState(Enum):
    CLOSED = "closed"       # Normal — calls pass through
    OPEN = "open"           # Failing — calls rejected immediately
    HALF_OPEN = "half_open" # Probing — one test call allowed


class CircuitBreaker:
    """
    Thread-safe sliding-window circuit breaker.

    State machine:
      CLOSED ──(N failures)──► OPEN ──(reset_timeout)──► HALF_OPEN
        ▲                                                      │
        └──────────────── (probe success) ────────────────────┘
                          (probe failure) ──────────────────► OPEN

    Usage:
        breaker = CircuitBreaker("gemini", failure_threshold=5, reset_timeout=60)

        try:
            result = breaker.call(my_fn, arg1, arg2)
        except CircuitOpenError:
            # Use fallback
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

        self._state = _CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_ts: float = 0.0
        self._lock = threading.Lock()

        log.info(
            "CircuitBreaker initialised",
            extra={
                "service": service_name,
                "threshold": failure_threshold,
                "reset_timeout_s": reset_timeout,
            },
        )

    # ── Internal helpers (always called under self._lock) ─────────────────────

    def _check_transition(self) -> bool:
        """Return True if a call should proceed. May transition OPEN→HALF_OPEN."""
        if self._state == _CircuitState.CLOSED:
            return True
        if self._state == _CircuitState.OPEN:
            if time.time() - self._last_failure_ts >= self.reset_timeout:
                self._state = _CircuitState.HALF_OPEN
                log.info(
                    "CircuitBreaker → HALF_OPEN (probing)",
                    extra={"service": self.service_name},
                )
                return True
            return False
        # HALF_OPEN: allow exactly one probe
        return True

    def _record_success(self) -> None:
        if self._state != _CircuitState.CLOSED:
            log.info("CircuitBreaker → CLOSED", extra={"service": self.service_name})
        self._state = _CircuitState.CLOSED
        self._failure_count = 0

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_ts = time.time()
        if (self._failure_count >= self.failure_threshold
                or self._state == _CircuitState.HALF_OPEN):
            self._state = _CircuitState.OPEN
            log.warning(
                "CircuitBreaker → OPEN",
                extra={
                    "service": self.service_name,
                    "failure_count": self._failure_count,
                    "reset_in_s": self.reset_timeout,
                },
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run fn through the circuit breaker guard."""
        with self._lock:
            if not self._check_transition():
                reset_in = self.reset_timeout - (time.time() - self._last_failure_ts)
                raise CircuitOpenError(self.service_name, max(reset_in, 0))

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result
        except (CircuitOpenError, LLMAuthError):
            raise  # Don't count these as circuit failures
        except Exception as exc:
            with self._lock:
                self._record_failure()
            raise exc

    def reset(self) -> None:
        """Manually reset the breaker (e.g., after a hotfix deployment)."""
        with self._lock:
            self._state = _CircuitState.CLOSED
            self._failure_count = 0
        log.info("CircuitBreaker manually reset", extra={"service": self.service_name})

    def get_status(self) -> dict:
        """Snapshot for health-check and metrics endpoints."""
        with self._lock:
            reset_in = 0.0
            if self._state == _CircuitState.OPEN:
                elapsed = time.time() - self._last_failure_ts
                reset_in = max(self.reset_timeout - elapsed, 0.0)
            return {
                "service": self.service_name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "reset_in_seconds": round(reset_in, 1),
            }

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._state == _CircuitState.OPEN


# ==============================================================================
# MODULE-LEVEL CIRCUIT BREAKERS  (singletons shared across graph nodes)
# ==============================================================================
from config import CIRCUIT_BREAKER_FAILURE_THRESHOLD, CIRCUIT_BREAKER_RESET_TIMEOUT

llm_breaker = CircuitBreaker(
    "gemini_llm",
    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT,
)

vectorstore_breaker = CircuitBreaker(
    "chromadb",
    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT,
)
