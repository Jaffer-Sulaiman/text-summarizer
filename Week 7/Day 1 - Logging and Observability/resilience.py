"""
resilience.py — Failure Handling & Resilience Primitives  (Week 7 Day 1)
=========================================================================
Extends Day 2 with richer structured logging at every resilience event:

NEW in Week 7 Day 1:
  - trace_id threaded into all log entries via an optional kwarg on invoke_with_retry
  - Retry log entries now include: attempt, max_retries, delay_s, error_type, error
  - Circuit breaker state transitions logged with from_state, to_state, failure_count
  - Non-retryable abort logged with error_type + full error string
  - classify_llm_error logs the classification decision for auditability
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
# EXCEPTION HIERARCHY  (unchanged from Day 2)
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


_NON_RETRYABLE = (LLMTimeoutError, LLMAuthError, CircuitOpenError)
_RETRYABLE = (LLMRateLimitError, LLMUnavailableError)


# ==============================================================================
# ERROR CLASSIFIER
# ==============================================================================

def classify_llm_error(exc: Exception) -> ResilienceError:
    """
    Map any raw exception from LangChain / Google GenAI SDK into our typed
    hierarchy by inspecting the error message string.
    """
    if isinstance(exc, ResilienceError):
        return exc

    msg = str(exc).lower()
    raw = str(exc)

    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        classified = LLMTimeoutError(f"LLM timed out: {raw}")
    elif "429" in msg or "quota" in msg or ("rate" in msg and "limit" in msg):
        classified = LLMRateLimitError(f"LLM quota/rate-limit: {raw}")
    elif any(k in msg for k in ("401", "403", "api key", "api_key",
                                 "authentication", "invalid key", "credentials")):
        classified = LLMAuthError(f"LLM auth failure: {raw}")
    elif any(k in msg for k in ("context", "token")) and any(
        k in msg for k in ("length", "window", "limit", "overflow", "too long")
    ):
        classified = LLMUnavailableError(f"Context window overflow: {raw}")
    elif any(k in msg for k in ("connection", "network", "unreachable",
                                  "connection refused", "eof", "broken pipe",
                                  "reset by peer", "socket")):
        classified = LLMUnavailableError(f"Network error: {raw}")
    else:
        classified = LLMUnavailableError(f"LLM call failed: {raw}")

    log.debug(
        "LLM error classified",
        extra={
            "original_error": raw[:200],
            "classified_as": type(classified).__name__,
            "error_code": classified.code,
        },
    )
    return classified


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
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            log.warning(
                "LLM call timed out",
                extra={
                    "timeout_seconds": timeout_seconds,
                    "fn": getattr(fn, "__name__", str(fn)),
                },
            )
            raise LLMTimeoutError(
                f"LLM did not respond within {timeout_seconds}s"
            )
        except Exception as exc:
            raise exc


# ==============================================================================
# 2. RETRY WITH EXPONENTIAL BACKOFF
# ==============================================================================

def invoke_with_retry(
    fn: Callable[..., T],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout_seconds: float = 30.0,
    trace_id: str = "",          # ★ NEW — threaded into log entries
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
            result = invoke_with_timeout(
                fn, *args,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )
            if attempt > 1:
                log.info(
                    "LLM call succeeded after retry",
                    extra={
                        "trace_id": trace_id,
                        "attempt": attempt,
                        "fn": getattr(fn, "__name__", str(fn)),
                    },
                )
            return result

        except _NON_RETRYABLE as exc:
            log.error(
                "Non-retryable LLM error — aborting immediately",
                extra={
                    "trace_id": trace_id,
                    "error_type": type(exc).__name__,
                    "error_code": exc.code,
                    "error": str(exc),
                    "attempt": attempt,
                    "max_retries": max_retries,
                },
            )
            raise

        except Exception as exc:
            typed = classify_llm_error(exc)

            if isinstance(typed, _NON_RETRYABLE):
                log.error(
                    "Non-retryable LLM error (classified) — aborting",
                    extra={
                        "trace_id": trace_id,
                        "error_type": type(typed).__name__,
                        "error_code": typed.code,
                        "error": str(typed),
                    },
                )
                raise typed

            last_error = typed

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))   # 1s, 2s, 4s …
                log.warning(
                    "LLM call failed — will retry",
                    extra={
                        "trace_id": trace_id,
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "retry_in_seconds": delay,
                        "error_type": type(typed).__name__,
                        "error_code": typed.code,
                        "error": str(typed)[:300],
                    },
                )
                time.sleep(delay)
            else:
                log.error(
                    "All LLM retries exhausted",
                    extra={
                        "trace_id": trace_id,
                        "attempts": max_retries,
                        "error_type": type(typed).__name__,
                        "last_error": str(typed)[:300],
                    },
                )

    raise RetryExhaustedError(
        f"LLM call failed after {max_retries} attempts. Last: {last_error}",
        attempts=max_retries,
    )


# ==============================================================================
# 3. CIRCUIT BREAKER
# ==============================================================================

class _CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Thread-safe sliding-window circuit breaker.

    State machine:
      CLOSED ──(N failures)──► OPEN ──(reset_timeout)──► HALF_OPEN
        ▲                                                      │
        └──────────────── (probe success) ────────────────────┘
                          (probe failure) ──────────────────► OPEN
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_transition(self) -> bool:
        """Return True if a call should proceed. May transition OPEN→HALF_OPEN."""
        if self._state == _CircuitState.CLOSED:
            return True
        if self._state == _CircuitState.OPEN:
            if time.time() - self._last_failure_ts >= self.reset_timeout:
                log.info(
                    "CircuitBreaker state transition",
                    extra={
                        "service": self.service_name,
                        "from_state": "open",
                        "to_state": "half_open",
                        "failure_count": self._failure_count,
                    },
                )
                self._state = _CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: allow exactly one probe
        return True

    def _record_success(self) -> None:
        prev = self._state.value
        if self._state != _CircuitState.CLOSED:
            log.info(
                "CircuitBreaker state transition",
                extra={
                    "service": self.service_name,
                    "from_state": prev,
                    "to_state": "closed",
                    "failure_count_reset": self._failure_count,
                },
            )
        self._state = _CircuitState.CLOSED
        self._failure_count = 0

    def _record_failure(self) -> None:
        prev = self._state.value
        self._failure_count += 1
        self._last_failure_ts = time.time()
        if (self._failure_count >= self.failure_threshold
                or self._state == _CircuitState.HALF_OPEN):
            self._state = _CircuitState.OPEN
            log.warning(
                "CircuitBreaker state transition",
                extra={
                    "service": self.service_name,
                    "from_state": prev,
                    "to_state": "open",
                    "failure_count": self._failure_count,
                    "failure_threshold": self.failure_threshold,
                    "reset_in_seconds": self.reset_timeout,
                },
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run fn through the circuit breaker guard."""
        with self._lock:
            if not self._check_transition():
                reset_in = self.reset_timeout - (time.time() - self._last_failure_ts)
                log.warning(
                    "CircuitBreaker rejected call — OPEN",
                    extra={
                        "service": self.service_name,
                        "state": "open",
                        "reset_in_seconds": round(max(reset_in, 0), 1),
                    },
                )
                raise CircuitOpenError(self.service_name, max(reset_in, 0))

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result
        except (CircuitOpenError, LLMAuthError):
            raise
        except Exception as exc:
            with self._lock:
                self._record_failure()
            raise exc

    def reset(self) -> None:
        """Manually reset the breaker (e.g., after a hotfix deployment)."""
        with self._lock:
            prev = self._state.value
            self._state = _CircuitState.CLOSED
            self._failure_count = 0
        log.info(
            "CircuitBreaker manually reset",
            extra={"service": self.service_name, "previous_state": prev},
        )

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
