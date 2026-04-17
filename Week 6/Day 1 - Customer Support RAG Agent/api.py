"""
api.py — FastAPI Application Layer
====================================
Production-grade REST API for the Logistics Customer Support RAG Agent.

Middleware stack:
  1. CORSMiddleware           — cross-origin requests (all origins for demo)
  2. RequestTimingMiddleware  — logs every request with method, path, status, duration
  3. RateLimitMiddleware      — sliding-window 60 req/min per client IP

Endpoints:
  GET  /health                   — Liveness + readiness probe
  GET  /metrics                  — Aggregate request stats
  POST /ingest                   — Upload + ingest document
  GET  /kb/documents             — List all ingested documents
  GET  /kb/stats                 — ChromaDB collection stats
  DELETE /kb/documents/{hash}    — Delete document by file_hash
  POST /chat                     — Main RAG chat endpoint
  DELETE /sessions/{session_id}  — Clear a conversation session
  GET  /sessions                 — List active sessions (admin)
"""

import time
import uuid
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
)
from ingestor import extract, IngestorError
from vectorstore import vector_manager
from memory import memory_store
from graph import rag_graph
from logger import get_logger

log = get_logger("api")

# ==============================================================================
# APP INITIALISATION
# ==============================================================================
app = FastAPI(
    title="SwiftShip Logistics Support — RAG API",
    description=(
        "Customer Support AI agent for SwiftShip Logistics. "
        "Supports document ingestion and grounded conversational Q&A."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ==============================================================================
# METRICS STORE  (in-memory, resets on restart)
# ==============================================================================
_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "total_chat_requests": 0,
    "total_ingest_requests": 0,
    "total_errors": 0,
    "latency_ms_samples": deque(maxlen=1000),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


# ==============================================================================
# MIDDLEWARE
# ==============================================================================

class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status code, and duration."""

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
    """
    Sliding-window rate limiter: RATE_LIMIT_RPM requests per minute per IP.
    Returns 429 when the limit is exceeded.
    """

    def __init__(self, app, rpm: int = RATE_LIMIT_RPM):
        super().__init__(app)
        self._rpm = rpm
        self._window: Dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for docs / openapi
        if request.url.path in ("/docs", "/redoc", "/openapi.json", "/health"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._window[client_ip]

        # Evict timestamps older than 60 seconds
        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self._rpm:
            log.warning("Rate limit exceeded", extra={"client_ip": client_ip})
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "detail": f"Maximum {self._rpm} requests per minute. Please slow down.",
                },
            )

        window.append(now)
        return await call_next(request)


# Register middleware (order matters — outermost first)
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
# AUTH DEPENDENCY  (optional — enabled only when API_KEY env is set)
# ==============================================================================
def verify_api_key(request: Request) -> None:
    """
    Validates X-API-Key header. Skipped entirely when API_KEY is not configured.
    Safe to use as a FastAPI dependency.
    """
    if not API_KEY:
        return  # Auth disabled in demo mode
    incoming = request.headers.get("X-API-Key", "")
    if incoming != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "code": "INVALID_API_KEY",
                "detail": "Provide a valid X-API-Key header.",
            },
        )


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH,
        description="Customer's question",
        examples=["What are the shipping zones for Zone 3?"],
    )
    session_id: str = Field(
        ...,
        min_length=8,
        max_length=64,
        description="Unique session identifier (UUID recommended)",
    )
    category_filter: Optional[str] = Field(
        default=None,
        description=f"Restrict retrieval to one category. Options: {DOCUMENT_CATEGORIES}",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[str]
    needs_escalation: bool
    confidence_score: float


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
    status: str
    kb_empty: bool
    active_sessions: int
    uptime_seconds: float
    version: str


# ==============================================================================
# HELPERS
# ==============================================================================

def _standardized_error(status: int, error: str, code: str, detail: str = None):
    content = {"error": error, "code": code}
    if detail:
        content["detail"] = detail
    return JSONResponse(status_code=status, content=content)


# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness + readiness probe. Returns 200 when the service is up."""
    uptime = time.time() - time.mktime(
        time.strptime(_metrics["started_at"], "%Y-%m-%dT%H:%M:%SZ")
    )
    return HealthResponse(
        status="ok",
        kb_empty=vector_manager.is_empty(),
        active_sessions=memory_store.active_count(),
        uptime_seconds=round(uptime, 1),
        version="1.0.0",
    )


@app.get("/metrics", tags=["System"])
async def get_metrics(_: None = Depends(verify_api_key)):
    """Return aggregate request metrics."""
    samples = list(_metrics["latency_ms_samples"])
    avg_latency = round(sum(samples) / len(samples), 2) if samples else 0.0
    return {
        "total_requests": _metrics["total_requests"],
        "total_chat_requests": _metrics["total_chat_requests"],
        "total_ingest_requests": _metrics["total_ingest_requests"],
        "total_errors": _metrics["total_errors"],
        "avg_latency_ms": avg_latency,
        "active_sessions": memory_store.active_count(),
        "started_at": _metrics["started_at"],
    }


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Knowledge Base"],
    dependencies=[Depends(verify_api_key)],
)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, TXT, or DOCX file to ingest"),
    category: str = "general",
):
    """
    Upload and ingest a document into the knowledge base.

    - Validates file type and size via the ingestor layer.
    - Chunks the text adaptively and stores in ChromaDB.
    - Rejects duplicate documents (same content hash).
    """
    _metrics["total_ingest_requests"] += 1

    # Validate category
    if category not in DOCUMENT_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Invalid category",
                "code": "INVALID_CATEGORY",
                "detail": f"Category must be one of: {DOCUMENT_CATEGORIES}",
            },
        )

    try:
        raw_bytes = await file.read()
    except Exception as e:
        _metrics["total_errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))

    # Extract text via ingestor
    try:
        payload = extract(file.filename, raw_bytes)
    except IngestorError as e:
        _metrics["total_errors"] += 1
        return _standardized_error(
            status=422,
            error=str(e),
            code=e.code,
        )

    # Ingest to vectorstore
    try:
        result = vector_manager.ingest_document(
            text=payload["text"],
            source_name=file.filename,
            category=category,
            page_count=payload["page_count"],
            file_type=payload["file_type"],
        )
    except Exception as e:
        _metrics["total_errors"] += 1
        log.error("Ingest failed", extra={"error": str(e)})
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Ingestion failed",
                "code": "INGEST_ERROR",
                "detail": str(e),
            },
        )

    if result["status"] == "error":
        _metrics["total_errors"] += 1
        return _standardized_error(
            status=422,
            error=result["message"],
            code=result.get("code", "INGEST_ERROR"),
        )

    if result["status"] == "duplicate":
        return JSONResponse(status_code=200, content=result)

    return IngestResponse(**result)


@app.get("/kb/documents", tags=["Knowledge Base"])
async def list_documents(_: None = Depends(verify_api_key)):
    """List all documents currently in the knowledge base."""
    docs = vector_manager.list_documents()
    return {"documents": docs, "count": len(docs)}


@app.get("/kb/stats", tags=["Knowledge Base"])
async def kb_stats(_: None = Depends(verify_api_key)):
    """Return collection-level stats for the knowledge base."""
    return vector_manager.get_stats()


@app.delete("/kb/documents/{file_hash}", tags=["Knowledge Base"])
async def delete_document(
    file_hash: str,
    _: None = Depends(verify_api_key),
):
    """Delete all chunks of a document identified by its MD5 hash."""
    result = vector_manager.delete_document(file_hash)
    if result["status"] == "not_found":
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Document not found",
                "code": "DOCUMENT_NOT_FOUND",
                "detail": f"No document with hash '{file_hash}' exists in the knowledge base.",
            },
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
    Main RAG chat endpoint.

    - Routes the query through the 10-node LangGraph pipeline.
    - Maintains per-session conversational memory.
    - Returns grounded answer, source citations, confidence score,
      and escalation flag.
    """
    _metrics["total_chat_requests"] += 1

    query = request.query.strip()

    # Guard: empty query after strip
    if not query:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Empty query",
                "code": "EMPTY_QUERY",
                "detail": "Query cannot be blank or whitespace only.",
            },
        )

    # Guard: knowledge base empty (only matters for non-greeting/non-off-topic queries,
    # but we warn upfront — the graph will handle gracefully regardless)
    kb_empty = vector_manager.is_empty()

    # Load session history
    history = memory_store.get_history(request.session_id)

    # Build graph input state
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
    }

    try:
        final_state = rag_graph.invoke(graph_input)
    except Exception as e:
        _metrics["total_errors"] += 1
        log.error("Graph execution failed", extra={"error": str(e), "session_id": request.session_id})
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AI pipeline execution failed",
                "code": "GRAPH_EXECUTION_ERROR",
                "detail": str(e),
            },
        )

    answer = final_state.get("answer", "An unexpected error occurred.")
    needs_escalation = final_state.get("needs_escalation", False)
    confidence_score = final_state.get("confidence_score", 0.5)
    clean_sources = list(set(final_state.get("clean_sources", [])))

    # Escalation when KB was empty and it was a logistics query
    if kb_empty and final_state.get("intent", "logistics") == "logistics":
        needs_escalation = True

    # Persist this turn to memory
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
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
async def clear_session(session_id: str, _: None = Depends(verify_api_key)):
    """Clear the conversation history for a given session."""
    deleted = memory_store.clear_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Session not found",
                "code": "SESSION_NOT_FOUND",
                "detail": f"No active session with ID '{session_id}'.",
            },
        )
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions", tags=["Sessions"])
async def list_sessions(_: None = Depends(verify_api_key)):
    """List all active sessions with metadata (admin endpoint)."""
    return {"sessions": memory_store.list_sessions(), "count": memory_store.active_count()}


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    log.info("Starting SwiftShip RAG API", extra={"host": API_HOST, "port": API_PORT})
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False, log_level="warning")
