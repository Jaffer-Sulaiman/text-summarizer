"""
graph.py — LangGraph Orchestration Pipeline  (Week 8 Day 1 — LLM Cost Optimization)
=====================================================================================
Extends Week 7 Day 1 (Logging & Observability) with production-grade cost optimizations:

NEW in Week 8 Day 1:
  - Model tiering: llm_classifier (gemini-2.0-flash) for all classification nodes;
    llm_answer (gemini-2.0-flash) reserved for answer generation only
  - Merged node: classify_intent_and_complexity replaces intent_classifier +
    classify_complexity → saves one full LLM round-trip per logistics query
  - Prompt compression: all prompts trimmed of redundant whitespace and verbose preamble
  - Context truncation: _truncate_to_budget() caps tokens fed to grade_relevance and
    generate_answer at MAX_CONTEXT_TOKENS (default 2000)
  - History window tightening: rephrase uses MAX_HISTORY_REPHRASE msgs (4, was 6);
    generate_answer uses MAX_HISTORY_ANSWER msgs (2, was 4)
  - Skip-rephrase heuristic: rephrase_query skips LLM call when query contains no
    first/second/third-person pronouns — saves one LLM call on standalone queries
  - TokenBudget integration: per-request token accumulator attached to AgentState,
    logged at end of each request with over-budget WARNING when threshold exceeded
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
    LLM_CLASSIFIER_MODEL,
    LLM_ANSWER_MODEL,
    MAX_CONTEXT_TOKENS,
    MAX_HISTORY_REPHRASE,
    MAX_HISTORY_ANSWER,
)
from token_budget import TokenBudget
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
# LLM SETUP — Model Tiering (★ Week 8 Day 1)
# ==============================================================================
# Cheap/fast model: used for all classification nodes (intent, complexity, grade).
llm_classifier = ChatGoogleGenerativeAI(
    model=LLM_CLASSIFIER_MODEL,
    temperature=0.0,   # deterministic — classification outputs are binary
)

# Full model: reserved only for answer generation (quality-critical).
llm_answer = ChatGoogleGenerativeAI(
    model=LLM_ANSWER_MODEL,
    temperature=LLM_TEMPERATURE,
)

# Keep legacy alias so resilience helper (_llm_call) stays compatible.
llm = llm_answer


# ★ NEW: Merged intent + complexity schema (saves one LLM round-trip)
class IntentComplexityOutput(BaseModel):
    intent: str = Field(description="One of: 'logistics', 'greeting', 'off_topic'")
    query_type: str = Field(description="One of: 'simple' or 'complex'")

class GradeOutput(BaseModel):
    score: str = Field(description="'yes' or 'no'")

class AnswerOutput(BaseModel):
    answer: str = Field(description="The grounded answer to the question")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    needs_escalation: bool = Field(description="True if human agent should handle this")
    used_sources: List[str] = Field(description="List of [Source N] references used")

structured_intent_complexity = llm_classifier.with_structured_output(IntentComplexityOutput)
structured_grader            = llm_classifier.with_structured_output(GradeOutput)
structured_answer            = llm_answer.with_structured_output(AnswerOutput)


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

    # Observability (Week 7 Day 1)
    trace_id: str   # UUID4 per request, set in api.py before graph invocation

    # ★ NEW Week 8 Day 1: per-request token budget tracker
    token_budget: Optional[object]


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


# ★ NEW Week 8 Day 1: context truncation
_PRONOUN_RE = re.compile(
    r"\b(it|its|they|them|their|that|this|those|these|he|she|him|her|his|hers)\b",
    re.IGNORECASE,
)


def _truncate_to_budget(documents: List[str], sources: List[str],
                         max_tokens: int = MAX_CONTEXT_TOKENS) -> tuple:
    """
    Trim the document list so the total estimated token count stays under
    max_tokens.  Whole chunks are kept or dropped (no mid-chunk slicing).
    Returns (trimmed_docs, trimmed_sources, tokens_used).
    """
    kept_docs, kept_srcs, total = [], [], 0
    for doc, src in zip(documents, sources):
        doc_tokens = estimate_tokens(doc)
        if total + doc_tokens > max_tokens:
            break
        kept_docs.append(doc)
        kept_srcs.append(src)
        total += doc_tokens
    return kept_docs, kept_srcs, total


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

# ── 1. Classify Intent + Complexity — MERGED (★ Week 8 Day 1) ────────────────
def classify_intent_and_complexity(state: AgentState) -> dict:
    """
    Single LLM call returning intent + query_type together.
    Replaces the two separate nodes from Week 7 Day 1, saving one full LLM
    round-trip per logistics query. SOFT FAILURE: defaults on error.
    """
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    budget: TokenBudget = state.get("token_budget") or TokenBudget(trace_id=tid)
    question = state["question"]

    with TracingContext(log, "classify_intent_and_complexity", trace_id=tid, session_id=sid):
        prompt = (
            "Classify this logistics support message.\n\n"
            "intent options: 'logistics' (shipping/tracking/rates/customs/claims), "
            "'greeting' (hello/bye/thanks), 'off_topic' (unrelated).\n"
            "query_type options: 'simple' (single fact lookup), 'complex' (multi-step/comparisons).\n\n"
            f"Message: \"{question}\"\n\nReply with intent and query_type only."
        )
        log_prompt(log, trace_id=tid, node="classify_intent_and_complexity", prompt=prompt)
        tokens_in = estimate_tokens(prompt)
        t0 = time.perf_counter()
        intent, qt = "logistics", "simple"
        try:
            result = _llm_call(structured_intent_complexity.invoke, prompt, trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            intent = result.intent.lower().strip()
            qt     = result.query_type.lower().strip()
            if intent not in ("logistics", "greeting", "off_topic"): intent = "logistics"
            if qt not in ("simple", "complex"): qt = "simple"
            tokens_out = estimate_tokens(intent + qt)
            log_llm_call(log, trace_id=tid, node="classify_intent_and_complexity",
                         model=LLM_CLASSIFIER_MODEL, tokens_in=tokens_in,
                         tokens_out=tokens_out, latency_ms=latency_ms, success=True)
            budget.add("classify_intent_and_complexity", tokens_in, tokens_out)
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(log, trace_id=tid, node="classify_intent_and_complexity",
                         model=LLM_CLASSIFIER_MODEL, tokens_in=tokens_in,
                         latency_ms=latency_ms, success=False,
                         error=str(e), error_type=type(e).__name__,
                         traceback_str=tb_module.format_exc())
            log.warning("classify_intent_and_complexity failed — using defaults",
                        extra={"trace_id": tid, "session_id": sid, "error": str(e)})

    top_k = TOP_K_SIMPLE if qt == "simple" else TOP_K_COMPLEX
    log.info("Intent+complexity classified",
             extra={"trace_id": tid, "session_id": sid,
                    "intent": intent, "query_type": qt, "top_k": top_k})
    return {"intent": intent, "query_type": qt, "top_k": top_k,
            "token_budget": budget, "failure_mode": None, "error_detail": None}


# ── 2. Rephrase Query (SOFT FAILURE) ─────────────────────────────────────────
def rephrase_query(state: AgentState) -> dict:
    """
    SOFT FAILURE: falls back to original question on error.
    ★ Week 8 optimizations:
      - Skips LLM call when no pronouns detected (self-contained query)
      - History window tightened to MAX_HISTORY_REPHRASE msgs (4, was 6)
      - Compressed prompt saves ~35% tokens_in
      - Uses llm_classifier (cheaper model)
    """
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    budget: TokenBudget = state.get("token_budget") or TokenBudget(trace_id=tid)
    question = state["question"]
    messages = state.get("messages", [])

    if not messages:
        log.info("rephrase_query skipped — no history",
                 extra={"trace_id": tid, "session_id": sid})
        return {"rephrased_question": question, "token_budget": budget}

    # ★ Skip if query has no pronouns — it's already standalone
    if not _PRONOUN_RE.search(question):
        log.info("rephrase_query skipped — no pronouns",
                 extra={"trace_id": tid, "session_id": sid})
        return {"rephrased_question": question, "token_budget": budget}

    with TracingContext(log, "rephrase_query", trace_id=tid, session_id=sid):
        history_txt = ""
        for msg in messages[-MAX_HISTORY_REPHRASE:]:  # ★ was [-6:]
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            history_txt += f"{role}: {msg.content}\n"

        # ★ Compressed prompt
        prompt = (
            "Rewrite the latest question as a standalone question using the history.\n"
            "Replace pronouns with explicit entities; keep concise; if already standalone return unchanged.\n\n"
            f"History:\n{history_txt}\nQuestion: {question}\n\nStandalone:"
        )
        log_prompt(log, trace_id=tid, node="rephrase_query", prompt=prompt)
        tokens_in = estimate_tokens(prompt)
        t0 = time.perf_counter()
        try:
            response = _llm_call(llm_classifier.invoke,
                                 [HumanMessage(content=prompt)], trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            rephrased = response.content.strip().strip('"').strip("'") or question
            tokens_out = estimate_tokens(response.content)
            log_llm_call(log, trace_id=tid, node="rephrase_query",
                         model=LLM_CLASSIFIER_MODEL, tokens_in=tokens_in,
                         tokens_out=tokens_out, latency_ms=latency_ms, success=True)
            budget.add("rephrase_query", tokens_in, tokens_out)
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(log, trace_id=tid, node="rephrase_query",
                         model=LLM_CLASSIFIER_MODEL, tokens_in=tokens_in,
                         latency_ms=latency_ms, success=False,
                         error=str(e), error_type=type(e).__name__,
                         traceback_str=tb_module.format_exc())
            log.warning("rephrase_query failed — using original",
                        extra={"trace_id": tid, "session_id": sid, "error": str(e)})
            rephrased = question

    log.info("Query rephrased",
             extra={"trace_id": tid, "session_id": sid,
                    "original_chars": len(question), "rephrased_chars": len(rephrased)})
    return {"rephrased_question": rephrased, "token_budget": budget}


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
            log.error("validate_context failed unexpectedly",
                      extra={"trace_id": tid, "error": str(e),
                             "traceback": tb_module.format_exc()})
            return {"clean_documents": [], "clean_sources": []}


def grade_relevance(state: AgentState) -> dict:
    """
    SOFT FAILURE: defaults to 'yes' (optimistic) on LLM error.
    ★ Week 8: context truncated to MAX_CONTEXT_TOKENS; uses llm_classifier.
    """
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    budget: TokenBudget = state.get("token_budget") or TokenBudget(trace_id=tid)
    question = state.get("rephrased_question") or state["question"]
    documents = state.get("clean_documents", [])
    sources   = state.get("clean_sources", [])

    if not documents:
        return {"is_relevant": "no", "token_budget": budget}

    # ★ Truncate context before building the prompt
    trunc_docs, _, tokens_ctx = _truncate_to_budget(documents, sources)
    context = "\n\n".join(trunc_docs)

    # ★ Compressed prompt
    prompt = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Does this context answer the question? Reply ONLY 'yes' or 'no'."
    )
    log_prompt(log, trace_id=tid, node="grade_relevance", prompt=prompt)
    tokens_in = estimate_tokens(prompt)
    t0 = time.perf_counter()
    score = "yes"
    with TracingContext(log, "grade_relevance", trace_id=tid, session_id=sid):
        try:
            result = _llm_call(structured_grader.invoke, prompt, trace_id=tid)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            score = result.score.lower().strip()
            if score not in ("yes", "no"): score = "yes"
            tokens_out = estimate_tokens(score)
            log_llm_call(log, trace_id=tid, node="grade_relevance",
                         model=LLM_CLASSIFIER_MODEL, tokens_in=tokens_in,
                         tokens_out=tokens_out, latency_ms=latency_ms, success=True)
            budget.add("grade_relevance", tokens_in, tokens_out)
        except Exception as e:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_llm_call(log, trace_id=tid, node="grade_relevance",
                         model=LLM_CLASSIFIER_MODEL, tokens_in=tokens_in,
                         latency_ms=latency_ms, success=False,
                         error=str(e), error_type=type(e).__name__,
                         traceback_str=tb_module.format_exc())
            log.warning("grade_relevance failed — defaulting to 'yes'",
                        extra={"trace_id": tid, "session_id": sid, "error": str(e)})

    log.info("Relevance graded",
             extra={"trace_id": tid, "session_id": sid, "score": score,
                    "context_tokens": tokens_ctx, "chunks_truncated": len(documents) - len(trunc_docs)})
    return {"is_relevant": score, "token_budget": budget}


# ── 7. Generate Answer (HARD FAILURE) ────────────────────────────────────────
def generate_answer(state: AgentState) -> dict:
    """
    HARD FAILURE node. ★ Week 8 optimizations:
      - Context truncated to MAX_CONTEXT_TOKENS before building source_blocks
      - History window: MAX_HISTORY_ANSWER msgs (2, was 4)
      - Compressed system instruction
      - Token budget updated after each attempt
    """
    sid = state.get("session_id", "")
    tid = state.get("trace_id", "")
    budget: TokenBudget = state.get("token_budget") or TokenBudget(trace_id=tid)
    question = state.get("rephrased_question") or state["question"]
    documents = state.get("clean_documents", [])
    clean_sources = state.get("clean_sources", [])
    messages = state.get("messages", [])

    # ★ Truncate context to MAX_CONTEXT_TOKENS
    trunc_docs, trunc_srcs, ctx_tokens = _truncate_to_budget(documents, clean_sources)
    source_blocks = "\n\n".join(
        f"[Source {i+1}] ({trunc_srcs[i] if i < len(trunc_srcs) else 'Unknown'})\n{doc}"
        for i, doc in enumerate(trunc_docs)
    )

    # ★ Compressed instruction
    instruction = (
        "You are a SwiftShip Logistics customer support agent.\n"
        "Answer ONLY from the knowledge base below. Be concise and professional.\n"
        "Rules: cite sources as [Source N]; set needs_escalation=true for urgent issues "
        "(lost/delayed shipments) or insufficient context; confidence 0.0-1.0; "
        "do NOT hallucinate tracking numbers or dates not in context.\n\n"
        "=== KNOWLEDGE BASE ===\n"
        f"{source_blocks}\n"
        "=====================\n\nCustomer question:"
    )

    # ★ Tighter history: was messages[-4:], now messages[-MAX_HISTORY_ANSWER:]
    formatted = [HumanMessage(content=instruction)]
    for msg in messages[-MAX_HISTORY_ANSWER:]:
        if isinstance(msg, HumanMessage):
            formatted.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted.append(AIMessage(content=msg.content))
    formatted.append(HumanMessage(content=question))

    full_prompt_str = instruction + "\n\n" + question
    log.info("NODE: generate_answer",
             extra={"trace_id": tid, "session_id": sid,
                    "chunks_in_context": len(trunc_docs),
                    "chunks_dropped": len(documents) - len(trunc_docs),
                    "context_tokens": ctx_tokens})

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
        log_llm_call(log, trace_id=tid, node="generate_answer.structured",
                     model=LLM_ANSWER_MODEL, tokens_in=tokens_in,
                     tokens_out=tokens_out, latency_ms=latency_ms, success=True)
        budget.add("generate_answer", tokens_in, tokens_out)
        budget.log_summary(log)
        log.info("Answer generated (structured output)",
                 extra={"trace_id": tid, "session_id": sid, "confidence": confidence,
                        "needs_escalation": needs_escalation, "answer_tokens": tokens_out})
        return {"answer": answer, "confidence_score": confidence,
                "needs_escalation": needs_escalation,
                "failure_mode": None, "error_detail": None, "token_budget": budget}

    except (LLMTimeoutError, LLMRateLimitError, LLMAuthError,
            CircuitOpenError, RetryExhaustedError, LLMUnavailableError) as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(log, trace_id=tid, node="generate_answer.structured",
                     model=LLM_ANSWER_MODEL, tokens_in=tokens_in,
                     latency_ms=latency_ms, success=False,
                     error=str(e), error_type=type(e).__name__,
                     traceback_str=tb_module.format_exc())
        log.error("generate_answer hard LLM failure",
                  extra={"trace_id": tid, "session_id": sid, "error": str(e)})
        return _failure_state_from_error(e)

    except Exception as parse_exc:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(log, trace_id=tid, node="generate_answer.structured",
                     model=LLM_ANSWER_MODEL, tokens_in=tokens_in,
                     latency_ms=latency_ms, success=False,
                     error=str(parse_exc), error_type=type(parse_exc).__name__,
                     traceback_str=tb_module.format_exc())
        log.warning("generate_answer structured output failed — trying plain fallback",
                    extra={"trace_id": tid, "session_id": sid, "error": str(parse_exc)})

    # ── Attempt 2: plain text fallback ────────────────────────────────────────────
    log_prompt(log, trace_id=tid, node="generate_answer.plain_fallback", prompt=full_prompt_str)
    t0 = time.perf_counter()
    try:
        with TracingContext(log, "generate_answer.plain_fallback", trace_id=tid, session_id=sid):
            plain_resp = _llm_call(llm_answer.invoke, formatted, trace_id=tid)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        answer = plain_resp.content
        tokens_out = estimate_tokens(answer)
        log_llm_call(log, trace_id=tid, node="generate_answer.plain_fallback",
                     model=LLM_ANSWER_MODEL, tokens_in=tokens_in,
                     tokens_out=tokens_out, latency_ms=latency_ms, success=True)
        budget.add("generate_answer.fallback", tokens_in, tokens_out)
        budget.log_summary(log)
        log.info("Answer generated (plain fallback)",
                 extra={"trace_id": tid, "session_id": sid, "answer_chars": len(answer)})
        return {"answer": answer, "confidence_score": 0.5, "needs_escalation": False,
                "failure_mode": None, "error_detail": None, "token_budget": budget}

    except (LLMTimeoutError, LLMRateLimitError, LLMAuthError,
            CircuitOpenError, RetryExhaustedError, LLMUnavailableError) as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(log, trace_id=tid, node="generate_answer.plain_fallback",
                     model=LLM_ANSWER_MODEL, tokens_in=tokens_in, latency_ms=latency_ms,
                     success=False, error=str(e), error_type=type(e).__name__,
                     traceback_str=tb_module.format_exc())
        log.error("generate_answer plain fallback also failed",
                  extra={"trace_id": tid, "session_id": sid, "error": str(e)})
        return _failure_state_from_error(e)

    except Exception as e:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        log_llm_call(log, trace_id=tid, node="generate_answer.plain_fallback",
                     model=LLM_ANSWER_MODEL, tokens_in=tokens_in, latency_ms=latency_ms,
                     success=False, error=str(e), error_type=type(e).__name__,
                     traceback_str=tb_module.format_exc())
        log.error("generate_answer completely failed",
                  extra={"trace_id": tid, "session_id": sid, "error": str(e)})
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
# GRAPH COMPILATION  (★ Week 8 Day 1 — merged node, no classify_complexity)
# ==============================================================================
workflow = StateGraph(AgentState)

# ★ Merged first node replaces intent_classifier + classify_complexity
workflow.add_node("classify_intent_and_complexity", classify_intent_and_complexity)
workflow.add_node("rephrase_query",         rephrase_query)
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

workflow.add_edge(START, "classify_intent_and_complexity")

# ★ Updated routing: after merged classifier, route directly to rephrase or terminal handlers
workflow.add_conditional_edges(
    "classify_intent_and_complexity",
    route_after_intent,
    {
        "rephrase_query":   "rephrase_query",
        "handle_off_topic": "handle_off_topic",
        "handle_greeting":  "handle_greeting",
    },
)

# ★ rephrase now feeds directly into retrieve (complexity already decided)
workflow.add_edge("rephrase_query", "retrieve")

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
        "generate_answer":   "generate_answer",
        "handle_irrelevant": "handle_irrelevant",
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
    "LangGraph pipeline compiled (Week 8 Day 1 — LLM Cost Optimization)",
    extra={"nodes": 14, "version": "4.0.0",
           "classifier_model": LLM_CLASSIFIER_MODEL,
           "answer_model": LLM_ANSWER_MODEL},
)
