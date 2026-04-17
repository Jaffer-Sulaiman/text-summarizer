"""
graph.py — LangGraph Orchestration Pipeline (10 nodes)
========================================================
Implements the core AI reasoning pipeline as a directed state graph.

Pipeline flow:
  START
    → intent_classifier          ← Logistics / greeting / off-topic guardrail
        ├── off_topic  ──► handle_off_topic  ──► END
        ├── greeting   ──► handle_greeting   ──► END
        └── logistics  ──►
    → rephrase_query             ← Make query standalone using chat history
    → classify_complexity        ← simple (k=3) vs complex (k=6)
    → retrieve                   ← ChromaDB similarity search
    → validate_context           ← Junk scrub, redaction, signal density
        └── all_bad ──► handle_degraded ──► END
    → grade_relevance            ← Is context actually useful?
        └── no ──► handle_irrelevant ──► END
    → generate_answer            ← Grounded response with source citations
        └── escalation ──► escalate_to_human ──► END
    → END

New vs Week 5:
  - intent_classifier:  domain guardrail for logistics context
  - rephrase_query:     standalone query reformulation using history
  - escalate_to_human:  detects low-confidence / complex cases
  - sources:            tracks source filenames for UI citation display
  - needs_escalation / confidence_score in state and output
"""

import re
import time
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import LLM_MODEL, LLM_TEMPERATURE, TOP_K_SIMPLE, TOP_K_COMPLEX
from vectorstore import vector_manager
from logger import get_logger, TimingContext

log = get_logger("graph")


# ==============================================================================
# LLM SETUP
# ==============================================================================
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------
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


structured_intent = llm.with_structured_output(IntentOutput)
structured_complexity = llm.with_structured_output(ComplexityOutput)
structured_grader = llm.with_structured_output(GradeOutput)
structured_answer = llm.with_structured_output(AnswerOutput)


# ==============================================================================
# STATE DEFINITION
# ==============================================================================
class AgentState(TypedDict):
    # Core identity
    session_id: str

    # Conversation memory (LangGraph managed)
    messages: Annotated[List[BaseMessage], add_messages]

    # Query processing
    question: str
    rephrased_question: str       # After standalone reformulation
    intent: str                   # logistics | greeting | off_topic

    # Retrieval
    query_type: str               # simple | complex
    top_k: int
    category_filter: Optional[str]
    documents: List[str]          # Raw chunks from ChromaDB
    sources: List[str]            # Source filenames per chunk

    # Context validation
    clean_documents: List[str]
    clean_sources: List[str]

    # Grading
    is_relevant: str              # yes | no

    # Output
    answer: str
    needs_escalation: bool
    confidence_score: float


# ==============================================================================
# CONTEXT VALIDATION HELPERS (hardened from Week 5)
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
    """Return (clean_chunks, clean_sources)."""
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
# GRAPH NODES
# ==============================================================================

# ── 1. Intent Classifier ─────────────────────────────────────────────────────
def intent_classifier(state: AgentState) -> dict:
    """
    Gate the conversation into three tracks:
      - 'logistics':  shipping, tracking, rates, claims, SLA, customs, etc.
      - 'greeting':   hi, hello, thanks, bye
      - 'off_topic':  anything clearly unrelated to logistics
    """
    log.info("NODE: intent_classifier", extra={"session_id": state.get("session_id", "")})
    question = state["question"]

    prompt = (
        "You are an intent classifier for a logistics company's customer support chatbot.\n\n"
        "Classify the following user message into EXACTLY ONE of these intents:\n"
        "  - 'logistics'  : question about shipping, tracking, delivery, rates, zones, "
        "customs, SLAs, claims, prohibited items, packaging, returns, or any logistics topic.\n"
        "  - 'greeting'   : a greeting, farewell, thank you, or social small-talk.\n"
        "  - 'off_topic'  : anything completely unrelated to logistics (e.g., weather, sports, "
        "cooking, general knowledge, politics).\n\n"
        f"User message: \"{question}\"\n\n"
        "Reply with only the intent label: 'logistics', 'greeting', or 'off_topic'."
    )

    try:
        result = structured_intent.invoke(prompt)
        intent = result.intent.lower().strip()
        if intent not in ("logistics", "greeting", "off_topic"):
            intent = "logistics"
    except Exception as e:
        log.warning("intent_classifier failed, defaulting to logistics", extra={"error": str(e)})
        intent = "logistics"

    log.info("Intent classified", extra={"intent": intent, "session_id": state.get("session_id", "")})
    return {"intent": intent}


# ── 2. Rephrase Query ─────────────────────────────────────────────────────────
def rephrase_query(state: AgentState) -> dict:
    """
    Rewrite the latest question into a standalone form that does not depend
    on pronouns or implicit references from conversation history.
    This ensures ChromaDB retrieval is always grounded in explicit terms.
    """
    log.info("NODE: rephrase_query", extra={"session_id": state.get("session_id", "")})
    question = state["question"]
    messages = state.get("messages", [])

    # If no prior history, nothing to rephrase
    if not messages:
        return {"rephrased_question": question}

    history_txt = ""
    for msg in messages[-6:]:  # last 3 turns (6 messages)
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        history_txt += f"{role}: {msg.content}\n"

    prompt = (
        "You are a query reformulation assistant for a logistics customer support system.\n\n"
        "Given the conversation history and the latest user question, rewrite the question "
        "as a STANDALONE question that can be understood without the history.\n\n"
        "Rules:\n"
        "- Replace pronouns (it, they, that, this) with the explicit entity they refer to.\n"
        "- Keep the question concise and preserving the original intent.\n"
        "- If the question is already standalone, return it unchanged.\n\n"
        f"Conversation history:\n{history_txt}\n"
        f"Latest question: {question}\n\n"
        "Standalone question:"
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        rephrased = response.content.strip().strip('"').strip("'")
        if not rephrased:
            rephrased = question
    except Exception as e:
        log.warning("rephrase_query failed, using original", extra={"error": str(e)})
        rephrased = question

    log.info(
        "Query rephrased",
        extra={
            "original": question,
            "rephrased": rephrased,
            "session_id": state.get("session_id", ""),
        },
    )
    return {"rephrased_question": rephrased}


# ── 3. Classify Complexity ────────────────────────────────────────────────────
def classify_complexity(state: AgentState) -> dict:
    """Decide simple (k=3) vs complex (k=6) retrieval."""
    log.info("NODE: classify_complexity", extra={"session_id": state.get("session_id", "")})
    question = state.get("rephrased_question") or state["question"]

    prompt = (
        "Classify this logistics customer query as 'simple' or 'complex'.\n\n"
        "  'simple'  : single fact, one entity, one date, one status lookup.\n"
        "  'complex' : comparisons, multi-step, aggregations, involves "
        "multiple zones / rate tiers / SLA conditions.\n\n"
        f"Query: {question}\n\n"
        "Reply with only 'simple' or 'complex'."
    )

    try:
        result = structured_complexity.invoke(prompt)
        qt = result.query_type.lower().strip()
        if qt not in ("simple", "complex"):
            qt = "simple"
    except Exception as e:
        log.warning("classify_complexity failed, defaulting to simple", extra={"error": str(e)})
        qt = "simple"

    top_k = TOP_K_SIMPLE if qt == "simple" else TOP_K_COMPLEX
    log.info("Complexity classified", extra={"query_type": qt, "top_k": top_k})
    return {"query_type": qt, "top_k": top_k}


# ── 4. Retrieve ───────────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> dict:
    """Retrieve top_k chunks from ChromaDB, optionally filtered by category."""
    log.info("NODE: retrieve", extra={"session_id": state.get("session_id", "")})
    question = state.get("rephrased_question") or state["question"]
    top_k = state.get("top_k", TOP_K_SIMPLE)
    category_filter = state.get("category_filter")

    docs = vector_manager.retrieve(question, top_k=top_k, category_filter=category_filter)

    doc_strings = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]

    log.info("Retrieval complete", extra={"chunks_retrieved": len(docs)})
    return {"documents": doc_strings, "sources": sources}


# ── 5. Validate Context ───────────────────────────────────────────────────────
def validate_context(state: AgentState) -> dict:
    """Scrub junk, annotate redactions, filter low-signal chunks."""
    log.info("NODE: validate_context", extra={"session_id": state.get("session_id", "")})
    raw = state["documents"]
    srcs = state.get("sources", ["Unknown"] * len(raw))

    clean, clean_srcs = _validate_chunks(raw, srcs)

    log.info(
        "Context validated",
        extra={"chunks_in": len(raw), "chunks_out": len(clean)},
    )
    return {"clean_documents": clean, "clean_sources": clean_srcs}


# ── 6. Grade Relevance ────────────────────────────────────────────────────────
def grade_relevance(state: AgentState) -> dict:
    """Check if cleaned context can answer the question."""
    log.info("NODE: grade_relevance", extra={"session_id": state.get("session_id", "")})
    question = state.get("rephrased_question") or state["question"]
    documents = state.get("clean_documents", [])

    if not documents:
        return {"is_relevant": "no"}

    context = "\n\n".join(documents)
    prompt = (
        "You are a document relevance grader for a logistics support system.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Customer question: {question}\n\n"
        "Does the context contain information that can answer the customer's question? "
        "Reply with ONLY 'yes' or 'no'."
    )

    try:
        result = structured_grader.invoke(prompt)
        score = result.score.lower().strip()
        if score not in ("yes", "no"):
            score = "yes"
    except Exception as e:
        log.warning("grade_relevance failed, defaulting to yes", extra={"error": str(e)})
        score = "yes"

    log.info("Relevance graded", extra={"score": score})
    return {"is_relevant": score}


# ── 7. Generate Answer ────────────────────────────────────────────────────────
def generate_answer(state: AgentState) -> dict:
    """
    Generate a grounded, cited answer using cleaned context.
    Returns structured output including confidence and escalation flag.
    """
    log.info("NODE: generate_answer", extra={"session_id": state.get("session_id", "")})
    question = state.get("rephrased_question") or state["question"]
    documents = state.get("clean_documents", [])
    clean_sources = state.get("clean_sources", [])
    messages = state.get("messages", [])

    # Build numbered source blocks with filenames
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
        "3. If a customer seems frustrated or the query is time-sensitive (delays, lost shipments), "
        "set needs_escalation=true.\n"
        "4. Provide confidence from 0.0 to 1.0 based on how well the context answers the question.\n"
        "5. Reference sources as [Source N] in your answer.\n"
        "6. If the knowledge base lacks sufficient information, set needs_escalation=true and "
        "confidence below 0.5.\n"
        "7. Do NOT hallucinate tracking numbers, dates, or specific case details not in the context.\n\n"
        "Answer the following customer question:"
    )

    # Build message chain (no system role for Gemini compatibility)
    formatted = [HumanMessage(content=instruction)]

    # Inject last few history turns for context
    for msg in messages[-4:]:
        if isinstance(msg, HumanMessage):
            formatted.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted.append(AIMessage(content=msg.content))

    formatted.append(HumanMessage(content=question))

    try:
        with TimingContext(log, "generate_answer.llm_invoke", session_id=state.get("session_id", "")):
            result = structured_answer.invoke(formatted)
        answer = result.answer
        confidence = max(0.0, min(1.0, result.confidence))
        needs_escalation = result.needs_escalation
    except Exception as e:
        log.error("generate_answer structured output failed, falling back", extra={"error": str(e)})
        # Graceful fallback to plain text generation
        try:
            plain_resp = llm.invoke(formatted)
            answer = plain_resp.content
        except Exception as e2:
            answer = "I'm sorry, I encountered an error generating a response. Please try again or contact our support team."
            log.error("generate_answer fallback also failed", extra={"error": str(e2)})
        confidence = 0.5
        needs_escalation = False

    log.info(
        "Answer generated",
        extra={
            "confidence": confidence,
            "needs_escalation": needs_escalation,
            "session_id": state.get("session_id", ""),
        },
    )
    return {
        "answer": answer,
        "confidence_score": confidence,
        "needs_escalation": needs_escalation,
    }


# ── Terminal nodes ────────────────────────────────────────────────────────────

def handle_off_topic(state: AgentState) -> dict:
    log.info("NODE: handle_off_topic", extra={"session_id": state.get("session_id", "")})
    return {
        "answer": (
            "I'm SwiftShip's customer support assistant and can only help with logistics-related "
            "questions — such as shipment tracking, delivery rates, shipping zones, customs, "
            "and claims.\n\n"
            "Is there anything shipping or logistics related I can help you with today? 📦"
        ),
        "confidence_score": 1.0,
        "needs_escalation": False,
    }


def handle_greeting(state: AgentState) -> dict:
    log.info("NODE: handle_greeting", extra={"session_id": state.get("session_id", "")})
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
    log.info("NODE: handle_irrelevant", extra={"session_id": state.get("session_id", "")})
    return {
        "answer": (
            "I wasn't able to find specific information about your question in our knowledge base.\n\n"
            "Please try rephrasing your question, or contact our support team directly:\n"
            "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
        ),
        "confidence_score": 0.0,
        "needs_escalation": True,
    }


def handle_degraded(state: AgentState) -> dict:
    log.info("NODE: handle_degraded", extra={"session_id": state.get("session_id", "")})
    return {
        "answer": (
            "The retrieved documents appear to contain corrupted or incomplete content, "
            "so I cannot provide a reliable answer.\n\n"
            "Please verify the quality of uploaded documents and try again, or contact "
            "our support team at 📧 support@swiftship.com"
        ),
        "confidence_score": 0.0,
        "needs_escalation": True,
    }


def escalate_to_human(state: AgentState) -> dict:
    log.info("NODE: escalate_to_human", extra={"session_id": state.get("session_id", "")})
    original_answer = state.get("answer", "")
    escalation_notice = (
        "\n\n---\n"
        "⚠️ **Escalation Notice**: This query has been flagged for human review. "
        "A SwiftShip support specialist will follow up with you shortly.\n"
        "📧 support@swiftship.com | 📞 1-800-SWIFT-01"
    )
    return {"answer": original_answer + escalation_notice}


# ==============================================================================
# CONDITIONAL EDGES
# ==============================================================================

def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "logistics")
    if intent == "off_topic":
        return "handle_off_topic"
    if intent == "greeting":
        return "handle_greeting"
    return "rephrase_query"


def route_after_validation(state: AgentState) -> str:
    clean = state.get("clean_documents", [])
    if not clean:
        return "handle_degraded"
    return "grade_relevance"


def route_after_grading(state: AgentState) -> str:
    return "generate_answer" if state.get("is_relevant") == "yes" else "handle_irrelevant"


def route_after_generation(state: AgentState) -> str:
    return "escalate_to_human" if state.get("needs_escalation", False) else END


# ==============================================================================
# GRAPH COMPILATION
# ==============================================================================
workflow = StateGraph(AgentState)

# Register all nodes
workflow.add_node("intent_classifier", intent_classifier)
workflow.add_node("rephrase_query", rephrase_query)
workflow.add_node("classify_complexity", classify_complexity)
workflow.add_node("retrieve", retrieve)
workflow.add_node("validate_context", validate_context)
workflow.add_node("grade_relevance", grade_relevance)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("handle_off_topic", handle_off_topic)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("handle_irrelevant", handle_irrelevant)
workflow.add_node("handle_degraded", handle_degraded)
workflow.add_node("escalate_to_human", escalate_to_human)

# Wire edges
workflow.add_edge(START, "intent_classifier")

workflow.add_conditional_edges(
    "intent_classifier",
    route_after_intent,
    {
        "rephrase_query": "rephrase_query",
        "handle_off_topic": "handle_off_topic",
        "handle_greeting": "handle_greeting",
    },
)

workflow.add_edge("rephrase_query", "classify_complexity")
workflow.add_edge("classify_complexity", "retrieve")
workflow.add_edge("retrieve", "validate_context")

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
        "generate_answer": "generate_answer",
        "handle_irrelevant": "handle_irrelevant",
    },
)

workflow.add_conditional_edges(
    "generate_answer",
    route_after_generation,
    {
        "escalate_to_human": "escalate_to_human",
        END: END,
    },
)

workflow.add_edge("handle_off_topic", END)
workflow.add_edge("handle_greeting", END)
workflow.add_edge("handle_irrelevant", END)
workflow.add_edge("handle_degraded", END)
workflow.add_edge("escalate_to_human", END)

# Compile the graph
rag_graph = workflow.compile()

log.info("LangGraph pipeline compiled", extra={"nodes": 12})
