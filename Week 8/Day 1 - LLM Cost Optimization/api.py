"""
api.py — FastAPI Application Layer  (Week 8 Day 1 — LLM Cost Optimization)
===========================================================================
Extends Week 7 Day 1 with:
  - token_budget initialised per request, threaded into graph_input
  - MAX_TOKENS_PER_REQUEST surfaced in /metrics response
  - version bumped to 4.0.0
"""


import asyncio
import time
import traceback
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from config import (
    API_HOST,
    API_PORT,
    API_KEY,
    RATE_LIMIT_RPM,
    DOCUMENT_CATEGORIES,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    GRAPH_TIMEOUT_SECONDS,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RESET_TIMEOUT,
    LOG_LEVEL,
    LOG_DIR,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_BACKUP_COUNT,
    MAX_TOKENS_PER_REQUEST,
    LLM_CLASSIFIER_MODEL,
    LLM_ANSWER_MODEL,
)
from ingestor import extract, IngestorError
from vectorstore import vector_manager
from resilience import (
    CircuitBreaker,
    CircuitOpenError,
    VectorStoreError,
)
from memory import memory_store
from graph import rag_graph
from logger import get_logger
from observability import init_logging, new_trace_id, tail_log_file

# ---------------------------------------------------------------------------
# Initialise logging FIRST — before any other imports emit log lines
# ---------------------------------------------------------------------------
init_logging(
    log_dir=LOG_DIR,
    log_level=LOG_LEVEL,
    max_bytes=LOG_FILE_MAX_BYTES,
    backup_count=LOG_FILE_BACKUP_COUNT,
)

log = get_logger("api")


# ==============================================================================
# APP
# ==============================================================================
app = FastAPI(
    title="SwiftShip Logistics Support — RAG API v4",
    description=(
        "Customer Support AI agent for SwiftShip Logistics. "
        "Week 8 Day 1: LLM Cost Optimization — merged classification node, "
        "model tiering, context truncation, prompt compression, token budget tracking."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ==============================================================================
# METRICS  (in-memory, same as Day 2 + token counters)
# ==============================================================================
_metrics: Dict[str, Any] = {
    "total_requests":           0,
    "total_chat_requests":      0,
    "total_ingest_requests":    0,
    "total_errors":             0,
    "total_timeouts":           0,
    "total_circuit_open":       0,
    "total_tokens_in":          0,   # ★ NEW — cumulative estimated tokens sent to LLM
    "total_tokens_out":         0,   # ★ NEW — cumulative estimated tokens received
    "latency_ms_samples":       deque(maxlen=1000),
    "started_at":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


# ==============================================================================
# CIRCUIT BREAKER  (API-level)
# ==============================================================================
chat_circuit_breaker = CircuitBreaker(
    "chat_pipeline",
    failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    reset_timeout=CIRCUIT_BREAKER_RESET_TIMEOUT,
)


# ==============================================================================
# MIDDLEWARE
# ==============================================================================

class RequestTimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Generate a trace_id for non-chat requests (chat generates its own)
        req_trace_id = request.headers.get("X-Trace-Id", new_trace_id())
        start = time.perf_counter()
        _metrics["total_requests"] += 1
        try:
            response = await call_next(request)
        except Exception as exc:
            _metrics["total_errors"] += 1
            raise exc
        finally:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            _metrics["latency_ms_samples"].append(elapsed)

        log.info(
            "HTTP request",
            extra={
                "trace_id": req_trace_id,
                "method":   request.method,
                "path":     request.url.path,
                "status":   response.status_code,
                "duration_ms": elapsed,
                "client":   request.client.host if request.client else "unknown",
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rpm: int = RATE_LIMIT_RPM):
        super().__init__(app)
        self._rpm = rpm
        self._window: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/docs", "/redoc", "/openapi.json", "/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._window[client_ip]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self._rpm:
            log.warning(
                "Rate limit exceeded",
                extra={"client_ip": client_ip, "requests_in_window": len(window)},
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "code":  "RATE_LIMIT_EXCEEDED",
                    "detail": f"Maximum {self._rpm} requests per minute.",
                },
            )
        window.append(now)
        return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestTimingMiddleware)


# ==============================================================================
# AUTH
# ==============================================================================
def verify_api_key(request: Request) -> None:
    if not API_KEY:
        return
    incoming = request.headers.get("X-API-Key", "")
    if incoming != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "code": "INVALID_API_KEY",
                    "detail": "Provide a valid X-API-Key header."},
        )


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class ChatRequest(BaseModel):
    query:           str          = Field(..., min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH)
    session_id:      str          = Field(..., min_length=8, max_length=64)
    category_filter: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer:           str
    session_id:       str
    sources:          List[str]
    needs_escalation: bool
    confidence_score: float
    failure_mode:     Optional[str] = None
    trace_id:         str = ""      # ★ NEW — for support correlation


class IngestResponse(BaseModel):
    status:            str
    message:           str
    source_name:       Optional[str] = None
    file_hash:         Optional[str] = None
    category:          Optional[str] = None
    chunks_added:      int = 0
    chunks_discarded:  int = 0
    doc_type_detected: Optional[str] = None
    avg_chunk_tokens:  Optional[int] = None
    upload_ts:         Optional[str] = None


class ErrorResponse(BaseModel):
    error:  str
    code:   str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status:           str
    kb_empty:         bool
    kb_healthy:       bool
    active_sessions:  int
    uptime_seconds:   float
    version:          str
    circuit_breakers: Dict[str, Any]


# ==============================================================================
# HELPERS
# ==============================================================================

def _error(status: int, error: str, code: str, detail: str = None):
    content = {"error": error, "code": code}
    if detail:
        content["detail"] = detail
    return JSONResponse(status_code=status, content=content)


async def _invoke_graph_async(graph_input: dict) -> dict:
    """Run blocking LangGraph.invoke in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rag_graph.invoke, graph_input)


# ==============================================================================
# ENDPOINTS — System
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Deep readiness probe — ChromaDB + all circuit breaker states."""
    uptime = time.time() - time.mktime(
        time.strptime(_metrics["started_at"], "%Y-%m-%dT%H:%M:%SZ")
    )

    kb_health  = vector_manager.health_check()
    kb_healthy = kb_health.get("healthy", False)

    breakers = {"chat_pipeline": chat_circuit_breaker.get_status()}
    from resilience import llm_breaker, vectorstore_breaker
    breakers["gemini_llm"] = llm_breaker.get_status()
    breakers["chromadb"]   = vectorstore_breaker.get_status()

    any_open = any(b["state"] == "open" for b in breakers.values())
    status   = "ok" if (kb_healthy and not any_open) else "degraded"

    return HealthResponse(
        status=status,
        kb_empty=vector_manager.is_empty(),
        kb_healthy=kb_healthy,
        active_sessions=memory_store.active_count(),
        uptime_seconds=round(uptime, 1),
        version="3.0.0",
        circuit_breakers=breakers,
    )


@app.get("/metrics", tags=["System"])
async def get_metrics(_: None = Depends(verify_api_key)):
    samples     = list(_metrics["latency_ms_samples"])
    avg_latency = round(sum(samples) / len(samples), 2) if samples else 0.0
    return {
        "total_requests":           _metrics["total_requests"],
        "total_chat_requests":      _metrics["total_chat_requests"],
        "total_ingest_requests":    _metrics["total_ingest_requests"],
        "total_errors":             _metrics["total_errors"],
        "total_timeouts":           _metrics["total_timeouts"],
        "total_circuit_open":       _metrics["total_circuit_open"],
        "total_tokens_in":          _metrics["total_tokens_in"],
        "total_tokens_out":         _metrics["total_tokens_out"],
        "avg_latency_ms":           avg_latency,
        "active_sessions":          memory_store.active_count(),
        "started_at":               _metrics["started_at"],
        "log_dir":                  LOG_DIR,
        "log_level":                LOG_LEVEL,
        "max_tokens_per_request":   MAX_TOKENS_PER_REQUEST,   # ★ NEW
        "classifier_model":         LLM_CLASSIFIER_MODEL,     # ★ NEW
        "answer_model":             LLM_ANSWER_MODEL,         # ★ NEW
    }


@app.get("/breakers", tags=["System"])
async def circuit_breaker_status(_: None = Depends(verify_api_key)):
    """Live circuit breaker states for all services."""
    from resilience import llm_breaker, vectorstore_breaker
    return {
        "chat_pipeline": chat_circuit_breaker.get_status(),
        "gemini_llm":    llm_breaker.get_status(),
        "chromadb":      vectorstore_breaker.get_status(),
    }


@app.post("/breakers/reset", tags=["System"])
async def reset_circuit_breakers(_: None = Depends(verify_api_key)):
    """Manually reset all circuit breakers."""
    from resilience import llm_breaker, vectorstore_breaker
    chat_circuit_breaker.reset()
    llm_breaker.reset()
    vectorstore_breaker.reset()
    log.info("All circuit breakers manually reset")
    return {"status": "reset", "message": "All circuit breakers reset to CLOSED."}


@app.get("/logs", tags=["System"])
async def get_logs(
    last_n: int = Query(default=100, ge=1, le=5000, description="Number of log lines to return"),
    _: None = Depends(verify_api_key),
):
    """
    ★ NEW in Week 7 Day 1 — Return the last N lines from the rotating log file.

    Useful for:
      - Live debugging without SSH access to the server
      - Correlating trace_ids from ChatResponse with log entries
      - Monitoring error patterns and latency trends

    Returns:
        lines: list of raw JSON log strings
        count: number of lines returned
        log_dir: directory where log files are stored
    """
    lines = tail_log_file(LOG_DIR, last_n=last_n)
    log.info("Log lines requested", extra={"last_n": last_n, "returned": len(lines)})
    return {
        "lines":   lines,
        "count":   len(lines),
        "log_dir": LOG_DIR,
    }


# ==============================================================================
# ENDPOINTS — Knowledge Base
# ==============================================================================

@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
    dependencies=[Depends(verify_api_key)],
)
async def ingest_document(
    file: UploadFile = File(...),
    category: str = "general",
):
    _metrics["total_ingest_requests"] += 1
    trace_id = new_trace_id()   # ★ NEW — trace_id per ingest

    if category not in DOCUMENT_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail={"error": "Invalid category", "code": "INVALID_CATEGORY",
                    "detail": f"Must be one of: {DOCUMENT_CATEGORIES}"},
        )

    log.info(
        "Ingest request received",
        extra={
            "trace_id":  trace_id,
            "filename":  file.filename,
            "category":  category,
            "content_type": file.content_type,
        },
    )

    try:
        raw_bytes = await file.read()
    except Exception as e:
        _metrics["total_errors"] += 1
        log.error(
            "Ingest file read failed",
            extra={"trace_id": trace_id, "error": str(e), "traceback": traceback.format_exc()},
        )
        raise HTTPException(status_code=500, detail=str(e))

    try:
        payload = extract(file.filename, raw_bytes)
    except IngestorError as e:
        _metrics["total_errors"] += 1
        log.warning(
            "Ingest extraction failed",
            extra={"trace_id": trace_id, "error": str(e), "code": e.code},
        )
        return _error(422, str(e), e.code)

    try:
        result = vector_manager.ingest_document(
            text=payload["text"],
            source_name=file.filename,
            category=category,
            page_count=payload["page_count"],
            file_type=payload["file_type"],
            trace_id=trace_id,   # ★ NEW
        )
    except VectorStoreError as e:
        _metrics["total_errors"] += 1
        log.error(
            "Ingest VectorStoreError",
            extra={"trace_id": trace_id, "error": str(e), "traceback": traceback.format_exc()},
        )
        return _error(503, "Knowledge base is temporarily unavailable.", "VECTOR_STORE_ERROR", str(e))
    except Exception as e:
        _metrics["total_errors"] += 1
        log.error(
            "Ingest unexpected error",
            extra={"trace_id": trace_id, "error": str(e), "traceback": traceback.format_exc()},
        )
        raise HTTPException(status_code=500, detail={"error": "Ingestion failed",
                                                      "code": "INGEST_ERROR", "detail": str(e)})

    if result["status"] == "error":
        _metrics["total_errors"] += 1
        return _error(422, result["message"], result.get("code", "INGEST_ERROR"))

    if result["status"] == "duplicate":
        return JSONResponse(status_code=200, content=result)

    return IngestResponse(**result)


@app.get("/kb/documents", tags=["Knowledge Base"])
async def list_documents(_: None = Depends(verify_api_key)):
    docs = vector_manager.list_documents()
    return {"documents": docs, "count": len(docs)}


@app.get("/kb/stats", tags=["Knowledge Base"])
async def kb_stats(_: None = Depends(verify_api_key)):
    return vector_manager.get_stats()


@app.delete("/kb/documents/{file_hash}", tags=["Knowledge Base"])
async def delete_document(file_hash: str, _: None = Depends(verify_api_key)):
    result = vector_manager.delete_document(file_hash)
    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail={"error": "Document not found", "code": "DOCUMENT_NOT_FOUND",
                    "detail": f"No document with hash '{file_hash}'."},
        )
    return result


# ==============================================================================
# ENDPOINTS — Chat
# ==============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)],
)
async def chat(request: ChatRequest):
    """
    Main RAG chat endpoint.

    Week 7 Day 1 additions:
      - trace_id generated per request, injected into graph_input, returned in response
      - Token totals incremented from final_state
      - All error blocks include trace_id + traceback in structured log fields
    """
    _metrics["total_chat_requests"] += 1
    trace_id = new_trace_id()   # ★ NEW — one UUID4 per chat request
    query = request.query.strip()

    log.info(
        "Chat request received",
        extra={
            "trace_id":        trace_id,
            "session_id":      request.session_id,
            "query_chars":     len(query),
            "query_preview":   query[:100],
            "category_filter": request.category_filter,
        },
    )

    if not query:
        raise HTTPException(
            status_code=422,
            detail={"error": "Empty query", "code": "EMPTY_QUERY",
                    "detail": "Query cannot be blank or whitespace only."},
        )

    # Circuit breaker check
    if chat_circuit_breaker.is_open:
        _metrics["total_circuit_open"] += 1
        status = chat_circuit_breaker.get_status()
        log.warning(
            "Chat circuit OPEN — rejecting request",
            extra={
                "trace_id":    trace_id,
                "session_id":  request.session_id,
                "reset_in":    status["reset_in_seconds"],
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error":  "AI chat pipeline is temporarily unavailable",
                "code":   "CHAT_CIRCUIT_OPEN",
                "detail": (
                    f"The service has experienced repeated failures and is paused. "
                    f"Retry in {status['reset_in_seconds']:.0f}s. "
                    "Contact support@swiftship.com if this persists."
                ),
            },
        )

    kb_empty = vector_manager.is_empty()
    history  = memory_store.get_history(request.session_id, trace_id=trace_id)

    from token_budget import TokenBudget
    graph_input = {
        "session_id":         request.session_id,
        "question":           query,
        "rephrased_question": "",
        "intent":             "",
        "query_type":         "",
        "top_k":              3,
        "category_filter":    request.category_filter,
        "messages":           history,
        "documents":          [],
        "sources":            [],
        "clean_documents":    [],
        "clean_sources":      [],
        "is_relevant":        "",
        "answer":             "",
        "needs_escalation":   False,
        "confidence_score":   0.0,
        "failure_mode":       None,
        "error_detail":       None,
        "trace_id":           trace_id,
        "token_budget":       TokenBudget(trace_id=trace_id),  # ★ NEW
    }

    # Pipeline timeout + circuit breaker
    try:
        pipeline_start = time.perf_counter()
        final_state = await asyncio.wait_for(
            _invoke_graph_async(graph_input),
            timeout=GRAPH_TIMEOUT_SECONDS,
        )
        pipeline_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
        chat_circuit_breaker._record_success()   # noqa: SLF001

        log.info(
            "Pipeline completed",
            extra={
                "trace_id":      trace_id,
                "session_id":    request.session_id,
                "latency_ms":    pipeline_ms,
                "failure_mode":  final_state.get("failure_mode"),
                "confidence":    final_state.get("confidence_score", 0.0),
                "escalated":     final_state.get("needs_escalation", False),
            },
        )

    except asyncio.TimeoutError:
        _metrics["total_errors"]   += 1
        _metrics["total_timeouts"] += 1
        chat_circuit_breaker._record_failure()   # noqa: SLF001
        log.error(
            "Graph pipeline timed out",
            extra={
                "trace_id":   trace_id,
                "session_id": request.session_id,
                "timeout_s":  GRAPH_TIMEOUT_SECONDS,
            },
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error":  "AI pipeline timed out",
                "code":   "GRAPH_TIMEOUT",
                "detail": (
                    f"The request did not complete within {GRAPH_TIMEOUT_SECONDS}s. "
                    "Please try again. If this persists, contact support@swiftship.com"
                ),
                "trace_id": trace_id,
            },
        )

    except CircuitOpenError as e:
        _metrics["total_circuit_open"] += 1
        log.warning(
            "CircuitOpenError reached /chat handler",
            extra={"trace_id": trace_id, "error": str(e)},
        )
        raise HTTPException(
            status_code=503,
            detail={"error": str(e), "code": "CIRCUIT_OPEN",
                    "detail": "Try again later.", "trace_id": trace_id},
        )

    except Exception as e:
        _metrics["total_errors"] += 1
        chat_circuit_breaker._record_failure()   # noqa: SLF001
        log.error(
            "Graph execution failed",
            extra={
                "trace_id":   trace_id,
                "session_id": request.session_id,
                "error":      str(e),
                "traceback":  traceback.format_exc(),
            },
        )
        raise HTTPException(
            status_code=500,
            detail={"error": "AI pipeline execution failed",
                    "code": "GRAPH_EXECUTION_ERROR",
                    "detail": str(e), "trace_id": trace_id},
        )

    answer           = final_state.get("answer", "An unexpected error occurred.")
    needs_escalation = final_state.get("needs_escalation", False)
    confidence_score = final_state.get("confidence_score", 0.5)
    clean_sources    = list(set(final_state.get("clean_sources", [])))
    failure_mode     = final_state.get("failure_mode")

    if kb_empty and final_state.get("intent", "logistics") == "logistics":
        needs_escalation = True

    memory_store.add_turn(
        session_id=request.session_id,
        human_text=query,
        ai_text=answer,
        trace_id=trace_id,
    )

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        sources=clean_sources,
        needs_escalation=needs_escalation,
        confidence_score=round(confidence_score, 3),
        failure_mode=failure_mode,
        trace_id=trace_id,   # ★ NEW
    )


# ==============================================================================
# ENDPOINTS — Sessions
# ==============================================================================

@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def clear_session(session_id: str, _: None = Depends(verify_api_key)):
    deleted = memory_store.clear_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found", "code": "SESSION_NOT_FOUND",
                    "detail": f"No active session '{session_id}'."},
        )
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions", tags=["Sessions"])
async def list_sessions(_: None = Depends(verify_api_key)):
    return {"sessions": memory_store.list_sessions(), "count": memory_store.active_count()}


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    log.info(
        "Starting SwiftShip RAG API v4",
        extra={
            "host":               API_HOST,
            "port":               API_PORT,
            "log_level":          LOG_LEVEL,
            "log_dir":            LOG_DIR,
            "version":            "4.0.0",
            "classifier_model":   LLM_CLASSIFIER_MODEL,
            "answer_model":       LLM_ANSWER_MODEL,
        },
    )
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False, log_level="warning")
