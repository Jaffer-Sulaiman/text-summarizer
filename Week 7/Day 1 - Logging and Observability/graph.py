"""
graph.py — LangGraph Orchestration Pipeline  (Week 7 Day 1 — Logging & Observability)
=======================================================================================
Extends Day 2 with full-coverage observability at every graph node:

NEW in Week 7 Day 1:
  - AgentState carries trace_id — threaded through every node log entry
  - Every node wrapped in TracingContext for per-node latency measurement
  - log_prompt() called before each LLM invocation (truncated preview always; full
    text only when LOG_PROMPT=true env var is set)
  - log_llm_call() emits a structured LLM audit record (model, tokens_in,
    tokens_out, latency_ms, success/failure) after each LLM call
  - estimate_tokens() used to compute tokens_in before the call and tokens_out
    from the response content afterward
  - retrieve() logs query, top_k, category, chunk sizes via log_retrieval()
    (delegation to vectorstore.py which calls observability.log_retrieval())
  - All hard-failure exception blocks capture full tracebacks as structured fields
"""

import re
import time
import traceback as tb_module
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM_RETRY_BASE_DELAY,
    TOP_K_SIMPLE,
    TOP_K_COMPLEX,
)
from vectorstore import vector_manager
from resilience import (
    llm_breaker,
    invoke_with_retry,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMAuthError,
    LLMUnavailableError,
    RetryExhaustedError,
    CircuitOpenError,
    VectorStoreError,
    classify_llm_error,
)
from logger import get_logger
from observability import (
    TracingContext,
    log_llm_call,
    log_prompt,
    estimate_tokens,
    new_trace_id,
)

log = get_logger("graph")


# ==============================================================================
# LLM SETUP
# ==============================================================================
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
)

class IntentOutput(BaseModel):
    intent: str = Field(description="One of: 'logistics', 'greeting', 'off_topic'")

class ComplexityOutput(BaseModel):
    query_type: str = Field(description="One of: 'simple' or 'complex'")

class GradeOutput(BaseModel):
    score: str = Field(description="'yes' or 'no'")

class AnswerOutput(BaseModel):
    answer: str = Field(description="The grounded answer to the question")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    needs_escalation: bool = Field(description="True if human agent should handle this")
    used_sources: List[str] = Field(description="List of [Source N] references used")

structured_intent     = llm.with_structured_output(IntentOutput)
structured_complexity = llm.with_structured_output(ComplexityOutput)
structured_grader     = llm.with_structured_output(GradeOutput)
structured_answer     = llm.with_structured_output(AnswerOutput)


# ==============================================================================
# STATE DEFINITION  (extended with trace_id)
# ==============================================================================
class AgentState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], add_messages]

    # Query
    question: str
    rephrased_question: str
    intent: str

    # Retrieval
    query_type: str
    top_k: int
    category_filter: Optional[str]
    documents: List[str]
    sources: List[str]

    # Context
    clean_documents: List[str]
    clean_sources: List[str]

    # Grading
    is_relevant: str

    # Output
    answer: str
    needs_escalation: bool
    confidence_score: float

    # Failure tracking
    failure_mode: Optional[str]
    error_detail: Optional[str]

    # ★ NEW: Observability
    trace_id: str   # UUID4 per request, set in api.py before graph invocation


# ==============================================================================
# CONTEXT VALIDATION HELPERS  (unchanged)
# ==============================================================================
_REDACTION_RE = re.compile(
    r"\[REDACTED\]|\[CONFIDENTIAL\]|█{2,}|\*{3,}|<REDACTED>",
    re.IGNORECASE,
)
_JUNK_RE = re.compile(r"[\x00\ufffd]|[^a-zA-Z0-9\s]{3,}")


def _scrub(chunk: str) -> str:
    return _JUNK_RE.sub(" ", chunk).strip()

def _is_redacted(chunk: str) -> bool:
    return bool(_REDACTION_RE.search(chunk))

def _signal_ok(chunk: str) -> bool:
    if not chunk:
        return False
    return sum(c.isalpha() for c in chunk) / len(chunk) >= 0.40

def _validate_chunks(
    chunks: List[str], sources: List[str]
) -> tuple[List[str], List[str]]:
    clean_c, clean_s = [], []
    for chunk, src in zip(chunks, sources):
        scrubbed = _scrub(chunk)
        if _is_redacted(scrubbed):
            clean_c.append("[Note: This section contains redacted content and cannot be used.]")
            clean_s.append(src)
            continue
        if not _signal_ok(scrubbed):
            continue
        clean_c.append(scrubbed)
        clean_s.append(src)
    return clean_c, clean_s


# ==============================================================================
# RESILIENT LLM INVOKE HELPER
# ==============================================================================
def _llm_call(fn, *args, trace_id: str = "", **kwargs):
    """Execute an LLM call via circuit breaker + retry + timeout."""
    return llm_breaker.call(
        invoke_with_retry,
        fn,
        *args,
        max_retries=LLM_MAX_RETRIES,
        base_delay=LLM_RETRY_BASE_DELAY,
        timeout_seconds=LLM_TIMEOUT_SECONDS,
        trace_id=trace_id,
        **kwargs,
    )


def _failure_state_from_error(exc: Exception) -> dict:
    """Map a resilience exception to failure_mode + error_detail state fields."""
    from resilience import ResilienceError
    if not isinstance(exc, ResilienceError):
        exc = classify_llm_error(exc)

    failure_map = {
        "LLM_TIMEOUT":      "llm_timeout",
        "LLM_RATE_LIMITED": "llm_rate_limited",
        "LLM_AUTH_ERROR":   "llm_auth_error",
        "CIRCUIT_OPEN":     "llm_circuit_open",
        "RETRY_EXHAUSTED":  "llm_unavailable",
        "LLM_UNAVAILABLE":  "llm_unavailable",
    }
    fm = failure_map.get(exc.code, "llm_unavailable")
    return {"failure_mode": fm, "error_detail": str(exc)}


# ==============================================================================
# GRAPH NODES
# ==============================================================================

# ── 1. Intent Classifier (SOFT FAILURE) ──────────────────────────────────────
def intent_classifier(state: AgentState) -> dict:
    """SOFT FAILURE: on error defaults to 'logistics'."""
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    question = state["question"]

    with TracingContext(log, "intent_classifier", trace_id=tid, session_id=sid):
        prompt = (
            "You are an intent classifier for a logistics company's customer support chatbot.\n\n"
            "Classify the following user message into EXACTLY ONE of these intents:\n"
            "  - 'logistics'  : question about shipping, tracking, delivery, rates, zones, "
            "customs, SLAs, claims, prohibited items, packaging, returns, or any logistics topic.\n"
            "  - 'greeting'   : a greeting, farewell, thank you, or social small-talk.\n"
            "  - 'off_topic'  : anything completely unrelated to logistics.\n\n"
            f"User message: \"{question}\"\n\n"
            "Reply with only the intent label: 'logistics', 'greeting', or 'off_topic'."
        )

        log_prompt(log, trace_id=tid, node="intent_classifier", prompt=prompt)
        tokens_in = estimate_tokens(prompt)
        t0 = time.perf_counter()

        try:
            result = _llm_call(structured_intent.invoke, prompt, trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            intent = result.intent.lower().strip()
            if intent not in ("logistics", "greeting", "off_topic"):
                intent = "logistics"
            tokens_out = estimate_tokens(intent)
            log_llm_call(
                log, trace_id=tid, node="intent_classifier", model=LLM_MODEL,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, success=True,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(
                log, trace_id=tid, node="intent_classifier", model=LLM_MODEL,
                tokens_in=tokens_in, latency_ms=latency_ms, success=False,
                error=str(e), error_type=type(e).__name__,
                traceback_str=tb_module.format_exc(),
            )
            log.warning(
                "intent_classifier failed — defaulting to 'logistics'",
                extra={"trace_id": tid, "session_id": sid, "error": str(e)},
            )
            intent = "logistics"

    log.info("Intent classified", extra={"trace_id": tid, "session_id": sid, "intent": intent})
    return {"intent": intent, "failure_mode": None, "error_detail": None}


# ── 2. Rephrase Query (SOFT FAILURE) ─────────────────────────────────────────
def rephrase_query(state: AgentState) -> dict:
    """SOFT FAILURE: on error falls back to the original question."""
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    question = state["question"]
    messages = state.get("messages", [])

    if not messages:
        log.info(
            "rephrase_query skipped — no history",
            extra={"trace_id": tid, "session_id": sid},
        )
        return {"rephrased_question": question}

    with TracingContext(log, "rephrase_query", trace_id=tid, session_id=sid):
        history_txt = ""
        for msg in messages[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            history_txt += f"{role}: {msg.content}\n"

        prompt = (
            "You are a query reformulation assistant for a logistics customer support system.\n\n"
            "Given the conversation history and the latest user question, rewrite the question "
            "as a STANDALONE question that can be understood without the history.\n\n"
            "Rules:\n"
            "- Replace pronouns (it, they, that, this) with the explicit entity they refer to.\n"
            "- Keep the question concise and preserve the original intent.\n"
            "- If the question is already standalone, return it unchanged.\n\n"
            f"Conversation history:\n{history_txt}\n"
            f"Latest question: {question}\n\n"
            "Standalone question:"
        )

        log_prompt(log, trace_id=tid, node="rephrase_query", prompt=prompt)
        tokens_in = estimate_tokens(prompt)
        t0 = time.perf_counter()

        try:
            response = _llm_call(llm.invoke, [HumanMessage(content=prompt)], trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            rephrased = response.content.strip().strip('"').strip("'")
            if not rephrased:
                rephrased = question
            tokens_out = estimate_tokens(response.content)
            log_llm_call(
                log, trace_id=tid, node="rephrase_query", model=LLM_MODEL,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, success=True,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(
                log, trace_id=tid, node="rephrase_query", model=LLM_MODEL,
                tokens_in=tokens_in, latency_ms=latency_ms, success=False,
                error=str(e), error_type=type(e).__name__,
                traceback_str=tb_module.format_exc(),
            )
            log.warning(
                "rephrase_query failed — using original question",
                extra={"trace_id": tid, "session_id": sid, "error": str(e)},
            )
            rephrased = question

    log.info(
        "Query rephrased",
        extra={
            "trace_id": tid, "session_id": sid,
            "original_chars": len(question), "rephrased_chars": len(rephrased),
        },
    )
    return {"rephrased_question": rephrased}


# ── 3. Classify Complexity (SOFT FAILURE) ────────────────────────────────────
def classify_complexity(state: AgentState) -> dict:
    """SOFT FAILURE: defaults to 'simple' on error."""
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    question = state.get("rephrased_question") or state["question"]

    with TracingContext(log, "classify_complexity", trace_id=tid, session_id=sid):
        prompt = (
            "Classify this logistics customer query as 'simple' or 'complex'.\n"
            "  'simple'  : single fact, one entity, one date, one status lookup.\n"
            "  'complex' : comparisons, multi-step, multi-zone, aggregations.\n\n"
            f"Query: {question}\n\nReply with only 'simple' or 'complex'."
        )

        log_prompt(log, trace_id=tid, node="classify_complexity", prompt=prompt)
        tokens_in = estimate_tokens(prompt)
        t0 = time.perf_counter()

        try:
            result = _llm_call(structured_complexity.invoke, prompt, trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            qt = result.query_type.lower().strip()
            if qt not in ("simple", "complex"):
                qt = "simple"
            tokens_out = estimate_tokens(qt)
            log_llm_call(
                log, trace_id=tid, node="classify_complexity", model=LLM_MODEL,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, success=True,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(
                log, trace_id=tid, node="classify_complexity", model=LLM_MODEL,
                tokens_in=tokens_in, latency_ms=latency_ms, success=False,
                error=str(e), error_type=type(e).__name__,
                traceback_str=tb_module.format_exc(),
            )
            log.warning(
                "classify_complexity failed — defaulting to 'simple'",
                extra={"trace_id": tid, "session_id": sid, "error": str(e)},
            )
            qt = "simple"

    top_k = TOP_K_SIMPLE if qt == "simple" else TOP_K_COMPLEX
    log.info(
        "Complexity classified",
        extra={"trace_id": tid, "session_id": sid, "query_type": qt, "top_k": top_k},
    )
    return {"query_type": qt, "top_k": top_k}


# ── 4. Retrieve (HARD FAILURE) ────────────────────────────────────────────────
def retrieve(state: AgentState) -> dict:
    """HARD FAILURE node — sets failure_mode on any retrieval problem."""
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    question = state.get("rephrased_question") or state["question"]
    top_k = state.get("top_k", TOP_K_SIMPLE)
    category_filter = state.get("category_filter")

    with TracingContext(log, "retrieve", trace_id=tid, session_id=sid):
        try:
            # vectorstore.retrieve() now internally calls log_retrieval()
            docs = vector_manager.retrieve(
                question,
                top_k=top_k,
                category_filter=category_filter,
                trace_id=tid,
            )
        except VectorStoreError as e:
            log.error(
                "Retrieval hard failure — VectorStoreError",
                extra={
                    "trace_id": tid, "session_id": sid,
                    "error": str(e), "traceback": tb_module.format_exc(),
                },
            )
            return {
                "documents": [], "sources": [],
                "failure_mode": "retrieval_error",
                "error_detail": str(e),
            }
        except Exception as e:
            log.error(
                "Retrieval hard failure — unexpected error",
                extra={
                    "trace_id": tid, "session_id": sid,
                    "error": str(e), "traceback": tb_module.format_exc(),
                },
            )
            return {
                "documents": [], "sources": [],
                "failure_mode": "retrieval_error",
                "error_detail": f"Unexpected error: {e}",
            }

    if not docs:
        log.info(
            "Retrieval returned 0 results — empty_retrieval",
            extra={"trace_id": tid, "session_id": sid},
        )
        return {
            "documents": [], "sources": [],
            "failure_mode": "empty_retrieval",
            "error_detail": "No documents matched the query in the knowledge base.",
        }

    doc_strings = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    log.info(
        "Retrieval successful",
        extra={
            "trace_id": tid, "session_id": sid,
            "chunks_returned": len(docs),
            "sources": list(set(sources)),
        },
    )
    return {
        "documents": doc_strings,
        "sources": sources,
        "failure_mode": None,
        "error_detail": None,
    }


# ── 5. Validate Context (INTERNAL) ───────────────────────────────────────────
def validate_context(state: AgentState) -> dict:
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")

    with TracingContext(log, "validate_context", trace_id=tid, session_id=sid):
        try:
            raw = state["documents"]
            srcs = state.get("sources", ["Unknown"] * len(raw))
            clean, clean_srcs = _validate_chunks(raw, srcs)
            log.info(
                "Context validated",
                extra={
                    "trace_id": tid, "session_id": sid,
                    "chunks_in": len(raw), "chunks_out": len(clean),
                    "chunks_dropped": len(raw) - len(clean),
                },
            )
            return {"clean_documents": clean, "clean_sources": clean_srcs}
        except Exception as e:
            log.error(
                "validate_context failed unexpectedly",
                extra={
                    "trace_id": tid, "error": str(e),
                    "traceback": tb_module.format_exc(),
                },
            )
            return {"clean_documents": [], "clean_sources": []}


# ── 6. Grade Relevance (SOFT FAILURE) ────────────────────────────────────────
def grade_relevance(state: AgentState) -> dict:
    """SOFT FAILURE: defaults to 'yes' (optimistic) on LLM error."""
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    question = state.get("rephrased_question") or state["question"]
    documents = state.get("clean_documents", [])

    if not documents:
        return {"is_relevant": "no"}

    with TracingContext(log, "grade_relevance", trace_id=tid, session_id=sid):
        context = "\n\n".join(documents)
        prompt = (
            "You are a document relevance grader for a logistics support system.\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"Customer question: {question}\n\n"
            "Does the context contain information that can answer the customer's question? "
            "Reply with ONLY 'yes' or 'no'."
        )

        log_prompt(log, trace_id=tid, node="grade_relevance", prompt=prompt)
        tokens_in = estimate_tokens(prompt)
        t0 = time.perf_counter()

        try:
            result = _llm_call(structured_grader.invoke, prompt, trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            score = result.score.lower().strip()
            if score not in ("yes", "no"):
                score = "yes"
            tokens_out = estimate_tokens(score)
            log_llm_call(
                log, trace_id=tid, node="grade_relevance", model=LLM_MODEL,
                tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, success=True,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(
                log, trace_id=tid, node="grade_relevance", model=LLM_MODEL,
                tokens_in=tokens_in, latency_ms=latency_ms, success=False,
                error=str(e), error_type=type(e).__name__,
                traceback_str=tb_module.format_exc(),
            )
            log.warning(
                "grade_relevance failed — defaulting to 'yes'",
                extra={"trace_id": tid, "session_id": sid, "error": str(e)},
            )
            score = "yes"

    log.info(
        "Relevance graded",
        extra={"trace_id": tid, "session_id": sid, "score": score},
    )
    return {"is_relevant": score}


# ── 7. Generate Answer (HARD FAILURE) ────────────────────────────────────────
def generate_answer(state: AgentState) -> dict:
    """
    HARD FAILURE node.
    Attempt 1: structured output (AnswerOutput)
    Attempt 2: plain text fallback
    Both wrapped in circuit breaker + retry + full observability logging.
    """
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    question = state.get("rephrased_question") or state["question"]
    documents = state.get("clean_documents", [])
    clean_sources = state.get("clean_sources", [])
    messages = state.get("messages", [])

    source_blocks = "\n\n".join(
        f"[Source {i+1}] ({clean_sources[i] if i < len(clean_sources) else 'Unknown'})\n{doc}"
        for i, doc in enumerate(documents)
    )

    instruction = (
        "You are a professional customer support agent for SwiftShip Logistics. "
        "Your role is to help customers with shipping, tracking, rates, customs, claims, and related topics.\n\n"
        "You have access to the following retrieved knowledge base extracts:\n\n"
        "=== KNOWLEDGE BASE ===\n"
        f"{source_blocks}\n"
        "======================\n\n"
        "Instructions:\n"
        "1. Answer ONLY based on the provided knowledge base. Do not use outside knowledge.\n"
        "2. Be professional, empathetic, and concise — you are talking to a customer.\n"
        "3. If the query is time-sensitive (delays, lost shipments), set needs_escalation=true.\n"
        "4. Provide confidence from 0.0 to 1.0 based on how well the context answers.\n"
        "5. Reference sources as [Source N] in your answer.\n"
        "6. If the knowledge base is insufficient, set needs_escalation=true and confidence < 0.5.\n"
        "7. Do NOT hallucinate tracking numbers, dates, or details not in the context.\n\n"
        "Answer the following customer question:"
    )

    formatted = [HumanMessage(content=instruction)]
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            formatted.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted.append(AIMessage(content=msg.content))
    formatted.append(HumanMessage(content=question))

    # Build full prompt string for logging/token estimation
    full_prompt_str = instruction + "\n\n" + question

    log.info(
        "NODE: generate_answer",
        extra={
            "trace_id": tid, "session_id": sid,
            "chunks_in_context": len(documents),
            "total_context_chars": sum(len(d) for d in documents),
            "total_context_tokens": estimate_tokens(source_blocks),
        },
    )

    # ── Attempt 1: structured output ──────────────────────────────────────────
    log_prompt(log, trace_id=tid, node="generate_answer.structured", prompt=full_prompt_str)
    tokens_in = estimate_tokens(full_prompt_str)
    t0 = time.perf_counter()

    try:
        with TracingContext(log, "generate_answer.structured", trace_id=tid, session_id=sid):
            result = _llm_call(structured_answer.invoke, formatted, trace_id=tid)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        answer = result.answer
        confidence = max(0.0, min(1.0, result.confidence))
        needs_escalation = result.needs_escalation
        tokens_out = estimate_tokens(answer)

        log_llm_call(
            log, trace_id=tid, node="generate_answer.structured", model=LLM_MODEL,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, success=True,
        )
        log.info(
            "Answer generated (structured output)",
            extra={
                "trace_id": tid, "session_id": sid,
                "confidence": confidence,
                "needs_escalation": needs_escalation,
                "answer_chars": len(answer),
                "answer_tokens": tokens_out,
            },
        )
        return {
            "answer": answer,
            "confidence_score": confidence,
            "needs_escalation": needs_escalation,
            "failure_mode": None,
            "error_detail": None,
        }

    except (LLMTimeoutError, LLMRateLimitError, LLMAuthError,
            CircuitOpenError, RetryExhaustedError, LLMUnavailableError) as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(
            log, trace_id=tid, node="generate_answer.structured", model=LLM_MODEL,
            tokens_in=tokens_in, latency_ms=latency_ms, success=False,
            error=str(e), error_type=type(e).__name__,
            traceback_str=tb_module.format_exc(),
        )
        log.error(
            "generate_answer hard LLM failure",
            extra={
                "trace_id": tid, "session_id": sid,
                "error_type": type(e).__name__, "error": str(e),
            },
        )
        return _failure_state_from_error(e)

    except Exception as parse_exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(
            log, trace_id=tid, node="generate_answer.structured", model=LLM_MODEL,
            tokens_in=tokens_in, latency_ms=latency_ms, success=False,
            error=str(parse_exc), error_type=type(parse_exc).__name__,
            traceback_str=tb_module.format_exc(),
        )
        log.warning(
            "generate_answer structured output failed — trying plain fallback",
            extra={"trace_id": tid, "session_id": sid, "error": str(parse_exc)},
        )

    # ── Attempt 2: plain text fallback ────────────────────────────────────────
    log_prompt(log, trace_id=tid, node="generate_answer.plain_fallback", prompt=full_prompt_str)
    t0 = time.perf_counter()

    try:
        with TracingContext(log, "generate_answer.plain_fallback", trace_id=tid, session_id=sid):
            plain_resp = _llm_call(llm.invoke, formatted, trace_id=tid)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        answer = plain_resp.content
        tokens_out = estimate_tokens(answer)
        log_llm_call(
            log, trace_id=tid, node="generate_answer.plain_fallback", model=LLM_MODEL,
            tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms, success=True,
        )
        log.info(
            "Answer generated (plain fallback)",
            extra={"trace_id": tid, "session_id": sid, "answer_chars": len(answer)},
        )
        return {
            "answer": answer,
            "confidence_score": 0.5,
            "needs_escalation": False,
            "failure_mode": None,
            "error_detail": None,
        }

    except (LLMTimeoutError, LLMRateLimitError, LLMAuthError,
            CircuitOpenError, RetryExhaustedError, LLMUnavailableError) as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(
            log, trace_id=tid, node="generate_answer.plain_fallback", model=LLM_MODEL,
            tokens_in=tokens_in, latency_ms=latency_ms, success=False,
            error=str(e), error_type=type(e).__name__,
            traceback_str=tb_module.format_exc(),
        )
        log.error(
            "generate_answer plain fallback also failed",
            extra={"trace_id": tid, "session_id": sid, "error": str(e)},
        )
        return _failure_state_from_error(e)

    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(
            log, trace_id=tid, node="generate_answer.plain_fallback", model=LLM_MODEL,
            tokens_in=tokens_in, latency_ms=latency_ms, success=False,
            error=str(e), error_type=type(e).__name__,
            traceback_str=tb_module.format_exc(),
        )
        log.error(
            "generate_answer completely failed",
            extra={
                "trace_id": tid, "session_id": sid,
                "error": str(e), "traceback": tb_module.format_exc(),
            },
        )
        return _failure_state_from_error(e)


# ==============================================================================
# TERMINAL NODES — Domain responses  (unchanged logic; trace_id added to logs)
# ==============================================================================

def handle_off_topic(state: AgentState) -> dict:
    log.info("NODE: handle_off_topic", extra={
        "session_id": state.get("session_id", ""), "trace_id": state.get("trace_id", "")})
    return {
        "answer": (
            "I'm SwiftShip's customer support assistant and can only help with "
            "logistics-related questions — such as shipment tracking, delivery rates, "
            "shipping zones, customs, and claims.\n\n"
            "Is there anything shipping or logistics related I can help you with today? 📦"
        ),
        "confidence_score": 1.0,
        "needs_escalation": False,
    }


def handle_greeting(state: AgentState) -> dict:
    log.info("NODE: handle_greeting", extra={
        "session_id": state.get("session_id", ""), "trace_id": state.get("trace_id", "")})
    return {
        "answer": (
            "Hello! 👋 Welcome to SwiftShip Customer Support.\n\n"
            "I'm your AI support assistant. I can help you with:\n"
            "• 📦 Shipment tracking and status updates\n"
            "• 🗺️ Shipping zones and delivery timelines\n"
            "• 💰 Rates and pricing information\n"
            "• 📋 Customs and compliance questions\n"
            "• 🛠️ Claims and dispute assistance\n\n"
            "What can I help you with today?"
        ),
        "confidence_score": 1.0,
        "needs_escalation": False,
    }


def handle_irrelevant(state: AgentState) -> dict:
    log.info("NODE: handle_irrelevant", extra={
        "session_id": state.get("session_id", ""), "trace_id": state.get("trace_id", "")})
    return {
        "answer": (
            "I wasn't able to find specific information about your question in our "
            "knowledge base.\n\n"
            "Please try rephrasing your question, or contact our support team:\n"
            "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
        ),
        "confidence_score": 0.0,
        "needs_escalation": True,
    }


def handle_degraded(state: AgentState) -> dict:
    log.info("NODE: handle_degraded", extra={
        "session_id": state.get("session_id", ""), "trace_id": state.get("trace_id", "")})
    return {
        "answer": (
            "The retrieved documents appear corrupted or incomplete, so I cannot "
            "provide a reliable answer.\n\n"
            "Please verify the quality of uploaded documents and try again, or contact:\n"
            "📧 support@swiftship.com"
        ),
        "confidence_score": 0.0,
        "needs_escalation": True,
    }


def escalate_to_human(state: AgentState) -> dict:
    log.info("NODE: escalate_to_human", extra={
        "session_id": state.get("session_id", ""), "trace_id": state.get("trace_id", ""),
        "confidence_score": state.get("confidence_score", 0.0)})
    original = state.get("answer", "")
    notice = (
        "\n\n---\n"
        "⚠️ **Escalation Notice**: This query has been flagged for human review. "
        "A SwiftShip support specialist will follow up shortly.\n"
        "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
    )
    return {"answer": original + notice}


# ==============================================================================
# TERMINAL NODES — Failure responses
# ==============================================================================

_LLM_FAILURE_MESSAGES = {
    "llm_timeout": (
        "Our AI system is taking longer than expected to respond. "
        "This is usually temporary — please try again in a moment."
    ),
    "llm_rate_limited": (
        "Our AI system is currently at capacity due to high demand. "
        "Please wait about a minute and try your question again."
    ),
    "llm_auth_error": (
        "There is a configuration issue with our AI system. "
        "Our technical team has been notified and is working to resolve it."
    ),
    "llm_circuit_open": (
        "Our AI service has experienced repeated failures and is temporarily "
        "paused to prevent further issues. Please try again in about a minute."
    ),
    "llm_unavailable": (
        "Our AI service is temporarily unreachable due to a network issue. "
        "Please try again shortly."
    ),
}


def handle_llm_failure(state: AgentState) -> dict:
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    fm  = state.get("failure_mode", "llm_unavailable")
    detail = state.get("error_detail", "")
    log.error(
        "NODE: handle_llm_failure",
        extra={
            "trace_id": tid, "session_id": sid,
            "failure_mode": fm, "error_detail": detail,
        },
    )
    customer_msg = _LLM_FAILURE_MESSAGES.get(fm, _LLM_FAILURE_MESSAGES["llm_unavailable"])
    answer = (
        f"{customer_msg}\n\n"
        "In the meantime, please contact our support team directly:\n"
        "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
    )
    return {"answer": answer, "confidence_score": 0.0, "needs_escalation": True}


def handle_empty_retrieval(state: AgentState) -> dict:
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    log.info("NODE: handle_empty_retrieval", extra={"trace_id": tid, "session_id": sid})
    return {
        "answer": (
            "I searched our knowledge base but found no documents related to your question.\n\n"
            "This could mean:\n"
            "• No documents have been uploaded to the knowledge base yet\n"
            "• Your question may cover a topic not in our current documentation\n"
            "• Try a different keyword or rephrase your question\n\n"
            "You can also contact our support team:\n"
            "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
        ),
        "confidence_score": 0.0,
        "needs_escalation": True,
    }


def handle_retrieval_failure(state: AgentState) -> dict:
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    detail = state.get("error_detail", "")
    log.error(
        "NODE: handle_retrieval_failure",
        extra={"trace_id": tid, "session_id": sid, "error_detail": detail},
    )
    return {
        "answer": (
            "I'm unable to search our knowledge base right now due to a technical issue.\n\n"
            "Our team has been notified. Please try again in a few minutes, or contact us:\n"
            "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
        ),
        "confidence_score": 0.0,
        "needs_escalation": True,
    }


# ==============================================================================
# CONDITIONAL EDGE FUNCTIONS  (unchanged logic)
# ==============================================================================

def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "logistics")
    if intent == "off_topic":
        return "handle_off_topic"
    if intent == "greeting":
        return "handle_greeting"
    return "rephrase_query"


def route_after_retrieve(state: AgentState) -> str:
    fm = state.get("failure_mode")
    if fm == "empty_retrieval":
        return "handle_empty_retrieval"
    if fm == "retrieval_error":
        return "handle_retrieval_failure"
    return "validate_context"


def route_after_validation(state: AgentState) -> str:
    clean = state.get("clean_documents", [])
    if not clean:
        return "handle_degraded"
    return "grade_relevance"


def route_after_grading(state: AgentState) -> str:
    return "generate_answer" if state.get("is_relevant") == "yes" else "handle_irrelevant"


def route_after_generation(state: AgentState) -> str:
    if state.get("failure_mode"):
        return "handle_llm_failure"
    if state.get("needs_escalation", False):
        return "escalate_to_human"
    return END


# ==============================================================================
# GRAPH COMPILATION
# ==============================================================================
workflow = StateGraph(AgentState)

workflow.add_node("intent_classifier",      intent_classifier)
workflow.add_node("rephrase_query",         rephrase_query)
workflow.add_node("classify_complexity",    classify_complexity)
workflow.add_node("retrieve",               retrieve)
workflow.add_node("validate_context",       validate_context)
workflow.add_node("grade_relevance",        grade_relevance)
workflow.add_node("generate_answer",        generate_answer)

workflow.add_node("handle_off_topic",       handle_off_topic)
workflow.add_node("handle_greeting",        handle_greeting)
workflow.add_node("handle_irrelevant",      handle_irrelevant)
workflow.add_node("handle_degraded",        handle_degraded)
workflow.add_node("escalate_to_human",      escalate_to_human)

workflow.add_node("handle_llm_failure",       handle_llm_failure)
workflow.add_node("handle_empty_retrieval",   handle_empty_retrieval)
workflow.add_node("handle_retrieval_failure", handle_retrieval_failure)

workflow.add_edge(START, "intent_classifier")

workflow.add_conditional_edges(
    "intent_classifier",
    route_after_intent,
    {
        "rephrase_query":   "rephrase_query",
        "handle_off_topic": "handle_off_topic",
        "handle_greeting":  "handle_greeting",
    },
)

workflow.add_edge("rephrase_query", "classify_complexity")
workflow.add_edge("classify_complexity", "retrieve")

workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {
        "validate_context":         "validate_context",
        "handle_empty_retrieval":   "handle_empty_retrieval",
        "handle_retrieval_failure": "handle_retrieval_failure",
    },
)

workflow.add_conditional_edges(
    "validate_context",
    route_after_validation,
    {
        "grade_relevance": "grade_relevance",
        "handle_degraded": "handle_degraded",
    },
)

workflow.add_conditional_edges(
    "grade_relevance",
    route_after_grading,
    {
        "generate_answer":  "generate_answer",
        "handle_irrelevant":"handle_irrelevant",
    },
)

workflow.add_conditional_edges(
    "generate_answer",
    route_after_generation,
    {
        "handle_llm_failure": "handle_llm_failure",
        "escalate_to_human":  "escalate_to_human",
        END:                  END,
    },
)

for node in (
    "handle_off_topic", "handle_greeting", "handle_irrelevant",
    "handle_degraded", "escalate_to_human",
    "handle_llm_failure", "handle_empty_retrieval", "handle_retrieval_failure",
):
    workflow.add_edge(node, END)

rag_graph = workflow.compile()

log.info(
    "LangGraph pipeline compiled (Week 7 Day 1 — Logging & Observability)",
    extra={"nodes": 15, "version": "3.0.0"},
)
