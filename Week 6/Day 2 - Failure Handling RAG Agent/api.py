"""
api.py — FastAPI Application Layer  (Day 2 — Failure Handling)
===============================================================
Extends Day 1 with:
  - Async graph timeout (asyncio.wait_for + GRAPH_TIMEOUT_SECONDS) → HTTP 504
  - Chat circuit breaker (chat_breaker) → HTTP 503 when graph keeps crashing
  - Deep /health endpoint that probes ChromaDB + circuit breaker states
  - failure_mode forwarded in ChatResponse for UI differentiation
  - HTTP 503 (Service Unavailable) distinct from 500 (Internal Error)
"""

import asyncio
import time
import traceback
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
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

log = get_logger("api")


# ==============================================================================
# APP
# ==============================================================================
app = FastAPI(
    title="SwiftShip Logistics Support — RAG API v2",
    description=(
        "Customer Support AI agent for SwiftShip Logistics. "
        "Day 2: Full failure handling — LLM timeout, retrieval errors, "
        "circuit breakers, and pipeline timeouts."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ==============================================================================
# METRICS
# ==============================================================================
_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "total_chat_requests": 0,
    "total_ingest_requests": 0,
    "total_errors": 0,
    "total_timeouts": 0,
    "total_circuit_open": 0,
    "latency_ms_samples": deque(maxlen=1000),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


# ==============================================================================
# CIRCUIT BREAKER  (API-level, wraps graph invocations)
# ==============================================================================
# If the full graph fails CIRCUIT_BREAKER_FAILURE_THRESHOLD times in a row,
# the API stops invoking the graph and returns HTTP 503 immediately.
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
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": elapsed,
                "client": request.client.host if request.client else "unknown",
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
            log.warning("Rate limit exceeded", extra={"client_ip": client_ip})
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED",
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
    query: str = Field(..., min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH)
    session_id: str = Field(..., min_length=8, max_length=64)
    category_filter: Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[str]
    needs_escalation: bool
    confidence_score: float
    failure_mode: Optional[str] = None    # NEW: surfaced to UI for error type display


class IngestResponse(BaseModel):
    status: str
    message: str
    source_name: Optional[str] = None
    file_hash: Optional[str] = None
    category: Optional[str] = None
    chunks_added: int = 0
    chunks_discarded: int = 0
    doc_type_detected: Optional[str] = None
    avg_chunk_tokens: Optional[int] = None
    upload_ts: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str                          # "ok" | "degraded" | "down"
    kb_empty: bool
    kb_healthy: bool                     # NEW: ChromaDB reachability
    active_sessions: int
    uptime_seconds: float
    version: str
    circuit_breakers: Dict[str, Any]     # NEW: breaker states


# ==============================================================================
# HELPERS
# ==============================================================================

def _error(status: int, error: str, code: str, detail: str = None):
    content = {"error": error, "code": code}
    if detail:
        content["detail"] = detail
    return JSONResponse(status_code=status, content=content)


async def _invoke_graph_async(graph_input: dict) -> dict:
    """Run blocking LangGraph.invoke in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rag_graph.invoke, graph_input)


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Deep readiness probe (NEW in Day 2):
    - Checks ChromaDB reachability via vector_manager.health_check()
    - Reports circuit breaker states for LLM and ChromaDB
    - Returns "degraded" if any subsystem has issues (not "down" = API is running)
    """
    uptime = time.time() - time.mktime(
        time.strptime(_metrics["started_at"], "%Y-%m-%dT%H:%M:%SZ")
    )

    kb_health = vector_manager.health_check()
    kb_healthy = kb_health.get("healthy", False)

    breakers = {
        "chat_pipeline":  chat_circuit_breaker.get_status(),
    }

    # Import module-level breakers from resilience
    from resilience import llm_breaker, vectorstore_breaker
    breakers["gemini_llm"] = llm_breaker.get_status()
    breakers["chromadb"]   = vectorstore_breaker.get_status()

    any_open = any(b["state"] == "open" for b in breakers.values())
    status = "ok" if (kb_healthy and not any_open) else "degraded"

    return HealthResponse(
        status=status,
        kb_empty=vector_manager.is_empty(),
        kb_healthy=kb_healthy,
        active_sessions=memory_store.active_count(),
        uptime_seconds=round(uptime, 1),
        version="2.0.0",
        circuit_breakers=breakers,
    )


@app.get("/metrics", tags=["System"])
async def get_metrics(_: None = Depends(verify_api_key)):
    samples = list(_metrics["latency_ms_samples"])
    avg_latency = round(sum(samples) / len(samples), 2) if samples else 0.0
    return {
        "total_requests":       _metrics["total_requests"],
        "total_chat_requests":  _metrics["total_chat_requests"],
        "total_ingest_requests":_metrics["total_ingest_requests"],
        "total_errors":         _metrics["total_errors"],
        "total_timeouts":       _metrics["total_timeouts"],       # NEW
        "total_circuit_open":   _metrics["total_circuit_open"],   # NEW
        "avg_latency_ms":       avg_latency,
        "active_sessions":      memory_store.active_count(),
        "started_at":           _metrics["started_at"],
    }


@app.get("/breakers", tags=["System"])
async def circuit_breaker_status(_: None = Depends(verify_api_key)):
    """NEW: Live circuit breaker states for all services."""
    from resilience import llm_breaker, vectorstore_breaker
    return {
        "chat_pipeline": chat_circuit_breaker.get_status(),
        "gemini_llm":    llm_breaker.get_status(),
        "chromadb":      vectorstore_breaker.get_status(),
    }


@app.post("/breakers/reset", tags=["System"])
async def reset_circuit_breakers(_: None = Depends(verify_api_key)):
    """NEW: Manually reset all circuit breakers (e.g., after a service restart)."""
    from resilience import llm_breaker, vectorstore_breaker
    chat_circuit_breaker.reset()
    llm_breaker.reset()
    vectorstore_breaker.reset()
    log.info("All circuit breakers manually reset")
    return {"status": "reset", "message": "All circuit breakers reset to CLOSED."}


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

    if category not in DOCUMENT_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail={"error": "Invalid category", "code": "INVALID_CATEGORY",
                    "detail": f"Must be one of: {DOCUMENT_CATEGORIES}"},
        )

    try:
        raw_bytes = await file.read()
    except Exception as e:
        _metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

    try:
        payload = extract(file.filename, raw_bytes)
    except IngestorError as e:
        _metrics["total_errors"] += 1
        return _error(422, str(e), e.code)

    try:
        result = vector_manager.ingest_document(
            text=payload["text"],
            source_name=file.filename,
            category=category,
            page_count=payload["page_count"],
            file_type=payload["file_type"],
        )
    except VectorStoreError as e:
        _metrics["total_errors"] += 1
        log.error("Ingest VectorStoreError", extra={"error": str(e)})
        return _error(503, "Knowledge base is temporarily unavailable.", "VECTOR_STORE_ERROR", str(e))
    except Exception as e:
        _metrics["total_errors"] += 1
        log.error("Ingest unexpected error", extra={"error": str(e)})
        traceback.print_exc()
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


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    dependencies=[Depends(verify_api_key)],
)
async def chat(request: ChatRequest):
    """
    Main RAG chat endpoint (Day 2 enhancements):
      1. Chat-pipeline circuit breaker → HTTP 503 if graph keeps crashing
      2. asyncio.wait_for timeout      → HTTP 504 if pipeline is too slow
      3. failure_mode in response      → UI can show specific error type
    """
    _metrics["total_chat_requests"] += 1
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail={"error": "Empty query", "code": "EMPTY_QUERY",
                    "detail": "Query cannot be blank or whitespace only."},
        )

    # ── Circuit breaker check (API level) ─────────────────────────────────────
    if chat_circuit_breaker.is_open:
        _metrics["total_circuit_open"] += 1
        status = chat_circuit_breaker.get_status()
        log.warning(
            "Chat circuit OPEN — rejecting request",
            extra={"reset_in": status["reset_in_seconds"], "session_id": request.session_id},
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI chat pipeline is temporarily unavailable",
                "code": "CHAT_CIRCUIT_OPEN",
                "detail": (
                    f"The service has experienced repeated failures and is paused. "
                    f"Retry in {status['reset_in_seconds']:.0f}s. "
                    "Contact support@swiftship.com if this persists."
                ),
            },
        )

    kb_empty = vector_manager.is_empty()
    history = memory_store.get_history(request.session_id)

    graph_input = {
        "session_id": request.session_id,
        "question": query,
        "rephrased_question": "",
        "intent": "",
        "query_type": "",
        "top_k": 3,
        "category_filter": request.category_filter,
        "messages": history,
        "documents": [],
        "sources": [],
        "clean_documents": [],
        "clean_sources": [],
        "is_relevant": "",
        "answer": "",
        "needs_escalation": False,
        "confidence_score": 0.0,
        "failure_mode": None,
        "error_detail": None,
    }

    # ── Pipeline timeout + circuit breaker ────────────────────────────────────
    try:
        final_state = await asyncio.wait_for(
            _invoke_graph_async(graph_input),
            timeout=GRAPH_TIMEOUT_SECONDS,
        )
        # Record success to chat circuit breaker
        chat_circuit_breaker._record_success()   # noqa: SLF001

    except asyncio.TimeoutError:
        _metrics["total_errors"] += 1
        _metrics["total_timeouts"] += 1
        chat_circuit_breaker._record_failure()   # noqa: SLF001
        log.error(
            "Graph pipeline timed out",
            extra={"timeout_s": GRAPH_TIMEOUT_SECONDS, "session_id": request.session_id},
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": "AI pipeline timed out",
                "code": "GRAPH_TIMEOUT",
                "detail": (
                    f"The request did not complete within {GRAPH_TIMEOUT_SECONDS}s. "
                    "Please try again. If this persists, contact support@swiftship.com"
                ),
            },
        )

    except CircuitOpenError as e:
        _metrics["total_circuit_open"] += 1
        raise HTTPException(
            status_code=503,
            detail={"error": str(e), "code": "CIRCUIT_OPEN",
                    "detail": "Try again later."},
        )

    except Exception as e:
        _metrics["total_errors"] += 1
        chat_circuit_breaker._record_failure()   # noqa: SLF001
        log.error(
            "Graph execution failed",
            extra={"error": str(e), "session_id": request.session_id},
        )
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "AI pipeline execution failed",
                    "code": "GRAPH_EXECUTION_ERROR",
                    "detail": str(e)},
        )

    answer          = final_state.get("answer", "An unexpected error occurred.")
    needs_escalation= final_state.get("needs_escalation", False)
    confidence_score= final_state.get("confidence_score", 0.5)
    clean_sources   = list(set(final_state.get("clean_sources", [])))
    failure_mode    = final_state.get("failure_mode")   # NEW

    if kb_empty and final_state.get("intent", "logistics") == "logistics":
        needs_escalation = True

    memory_store.add_turn(
        session_id=request.session_id,
        human_text=query,
        ai_text=answer,
    )

    return ChatResponse(
        answer=answer,
        session_id=request.session_id,
        sources=clean_sources,
        needs_escalation=needs_escalation,
        confidence_score=round(confidence_score, 3),
        failure_mode=failure_mode,
    )


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
    log.info("Starting SwiftShip RAG API v2", extra={"host": API_HOST, "port": API_PORT})
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False, log_level="warning")
