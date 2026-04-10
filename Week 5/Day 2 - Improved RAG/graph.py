import re
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from vectorstore import vector_manager

# ==============================================================================
# STATE DEFINITION
# ==============================================================================
class AgentState(TypedDict):
    """Full RAG graph state."""
    messages: Annotated[List[BaseMessage], add_messages]   # Conversational memory
    question: str
    query_type: str        # "simple" | "complex"
    top_k: int             # Dynamically decided retrieval count
    documents: List[str]   # Raw chunks from ChromaDB
    clean_documents: List[str]  # After context validation
    is_relevant: str
    answer: str

# ==============================================================================
# LLM CONFIGURATION
# ==============================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

# Structured graders
class GradeOutput(BaseModel):
    score: str = Field(description="Relevance score: 'yes' or 'no'")

class QueryTypeOutput(BaseModel):
    query_type: str = Field(description="Query complexity: 'simple' or 'complex'")

structured_grader = llm.with_structured_output(GradeOutput)
structured_classifier = llm.with_structured_output(QueryTypeOutput)

# ==============================================================================
# CONTEXT QUALITY VALIDATION HELPERS
# ==============================================================================
# Redaction markers to detect deliberately hidden content
_REDACTION_PATTERNS = re.compile(
    r"\[REDACTED\]|\[CONFIDENTIAL\]|█{2,}|\*{3,}|<REDACTED>",
    re.IGNORECASE,
)
# Junk: null bytes, replacement char, 3+ non-alphanumeric runs (----, ####, . . .)
_JUNK_PATTERN = re.compile(r"[\x00\ufffd]|[^a-zA-Z0-9\s]{3,}")


def _scrub_chunk(chunk: str) -> str:
    """Strip null bytes, replacement chars, and long non-alphanumeric runs."""
    return _JUNK_PATTERN.sub(" ", chunk).strip()


def _is_redacted(chunk: str) -> bool:
    """Return True if the chunk contains explicit redaction markers."""
    return bool(_REDACTION_PATTERNS.search(chunk))


def _signal_density_ok(chunk: str) -> bool:
    """
    Return True when at least 40 % of characters are alphabetic.
    Catches whitespace pages, separator lines, and footer debris.
    """
    if not chunk:
        return False
    alpha_ratio = sum(c.isalpha() for c in chunk) / len(chunk)
    return alpha_ratio >= 0.40


def validate_chunks(chunks: List[str]) -> List[str]:
    """
    Validate and clean each retrieved chunk:
    1. Scrub junk symbols.
    2. Replace redacted chunks with a transparent annotation.
    3. Discard chunks that fail the signal-density check.
    """
    validated = []
    for chunk in chunks:
        scrubbed = _scrub_chunk(chunk)

        if _is_redacted(scrubbed):
            # Make the redaction visible to the LLM so it doesn't hallucinate
            validated.append(
                "[Note: This section contains redacted or confidential content "
                "and cannot be used to answer the query.]"
            )
            continue

        if not _signal_density_ok(scrubbed):
            # Discard whitespace / separator debris entirely
            continue

        validated.append(scrubbed)

    return validated

# ==============================================================================
# NODES
# ==============================================================================

def classify_query(state: AgentState):
    """
    Classify the user query as 'simple' or 'complex'.
    - simple  → single-entity lookup   → top_k = 3
    - complex → multi-entity / compare → top_k = 6
    Both values are capped later inside vectorstore.retrieve().
    """
    print("---CLASSIFY QUERY---")
    question = state["question"]

    prompt = (
        "You are a query complexity classifier. "
        "Classify the following user question as EITHER 'simple' or 'complex'.\n\n"
        "Rules:\n"
        "- 'simple'  : asks for a single fact, name, date, or definition.\n"
        "- 'complex' : asks for comparisons, aggregations, multi-step reasoning, "
        "or information that likely spans multiple sections of a document.\n\n"
        f"Question: {question}\n\n"
        "Respond with ONLY the word 'simple' or 'complex'."
    )

    try:
        result = structured_classifier.invoke(prompt)
        qt = result.query_type.lower().strip()
        if qt not in ("simple", "complex"):
            qt = "simple"
    except Exception as e:
        print(f"Classifier error (defaulting to simple): {e}")
        qt = "simple"

    top_k = 3 if qt == "simple" else 6
    print(f"Query type: {qt}  →  top_k = {top_k}")
    return {"query_type": qt, "top_k": top_k}


def retrieve(state: AgentState):
    """Retrieve top_k chunks from ChromaDB (DB-size capping handled in vectorstore)."""
    print("---RETRIEVE---")
    question = state["question"]
    top_k = state.get("top_k", 4)
    docs = vector_manager.retrieve(question, top_k=top_k)
    doc_strings = [doc.page_content for doc in docs]
    return {"documents": doc_strings}


def validate_context(state: AgentState):
    """
    Clean and validate every retrieved chunk.
    Discards junk, annotates redactions, checks signal density.
    """
    print("---VALIDATE CONTEXT---")
    raw_chunks = state["documents"]
    clean = validate_chunks(raw_chunks)
    print(f"  chunks in: {len(raw_chunks)}  →  chunks out: {len(clean)}")
    return {"clean_documents": clean}


def grade_documents(state: AgentState):
    """Check if *cleaned* documents are relevant to the question."""
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state.get("clean_documents", [])

    if not documents:
        return {"is_relevant": "no"}

    context = "\n\n".join(documents)

    prompt = (
        "You are a document relevance grader.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Does the context contain information that can answer the question? "
        "Reply with ONLY 'yes' or 'no'."
    )

    try:
        result = structured_grader.invoke(prompt)
        score = result.score.lower().strip()
        if score not in ("yes", "no"):
            score = "yes"
    except Exception as e:
        print(f"Grader error (defaulting to yes): {e}")
        score = "yes"

    print(f"Relevance: {score}")
    return {"is_relevant": score}


def generate(state: AgentState):
    """
    Generate a grounded, cited answer using cleaned context + conversational history.

    Prompt design changes vs Day 1:
    1. Numbered [Source N] blocks so the LLM can cite exactly.
    2. Explicit role: 'precise Q&A assistant', not generic 'helpful assistant'.
    3. Chain-of-thought instruction before composing the answer.
    4. Structured output format: Answer + Evidence.
    5. All instructions embedded in first HumanMessage (no system role) to avoid
       convert_system_message_to_human corruption with Gemini.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state.get("clean_documents", [])
    messages = state.get("messages", [])

    # Build numbered source blocks
    source_blocks = "\n\n".join(
        f"[Source {i+1}]\n{doc}" for i, doc in enumerate(documents)
    )

    instruction_block = (
        "You are a precise document Q&A assistant. "
        "Your sole job is to answer questions using ONLY the retrieved document sources listed below. "
        "You must NEVER use outside knowledge or make assumptions beyond what is written.\n\n"
        "=== RETRIEVED SOURCES ===\n"
        f"{source_blocks}\n"
        "=========================\n\n"
        "Instructions:\n"
        "1. First, silently identify which source(s) contain relevant information.\n"
        "2. Compose a clear, concise answer grounded strictly in those sources.\n"
        "3. Structure your response EXACTLY as:\n"
        "   **Answer**: <your direct answer here>\n"
        "   **Evidence**: <brief quote or paraphrase from [Source N] that supports it>\n"
        "4. If no source answers the question, reply:\n"
        "   **Answer**: I cannot find this information in the provided documents.\n"
        "   **Evidence**: N/A\n"
        "5. Do NOT hallucinate, speculate, or introduce external facts.\n\n"
        "Do you understand? Good. Now answer the following question:\n"
    )

    # Build message list — all as HumanMessage/AIMessage, no system role
    formatted_messages = [HumanMessage(content=instruction_block)]

    # Inject prior conversational turns
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            formatted_messages.append(HumanMessage(content=msg.content))
        elif isinstance(msg, AIMessage):
            formatted_messages.append(AIMessage(content=msg.content))

    # Current question last
    formatted_messages.append(HumanMessage(content=question))

    response = llm.invoke(formatted_messages)
    return {"answer": response.content}


def handle_irrelevant(state: AgentState):
    """Decline out-of-scope queries gracefully."""
    print("---IRRELEVANT / OUT OF SCOPE---")
    return {
        "answer": (
            "I'm sorry, the documents in my knowledge base do not contain "
            "information relevant to your question. "
            "Please ask something related to the uploaded documents!"
        )
    }


def handle_degraded_context(state: AgentState):
    """All chunks were discarded by the context validator."""
    print("---DEGRADED CONTEXT---")
    return {
        "answer": (
            "The retrieved content was too degraded, redacted, or corrupted "
            "to produce a reliable answer. Please verify the uploaded document quality."
        )
    }

# ==============================================================================
# CONDITIONAL EDGES
# ==============================================================================

def route_after_validation(state: AgentState):
    """If all chunks were cleaned away, short-circuit to degraded handler."""
    clean = state.get("clean_documents", [])
    if not clean:
        print("  → All chunks removed by validator. Routing to degraded handler.")
        return "handle_degraded_context"
    return "grade_documents"


def route_after_grading(state: AgentState):
    return "generate" if state["is_relevant"] == "yes" else "handle_irrelevant"

# ==============================================================================
# GRAPH COMPILATION
# ==============================================================================
workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("classify_query", classify_query)
workflow.add_node("retrieve", retrieve)
workflow.add_node("validate_context", validate_context)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("handle_irrelevant", handle_irrelevant)
workflow.add_node("handle_degraded_context", handle_degraded_context)

# Wire edges
workflow.add_edge(START, "classify_query")
workflow.add_edge("classify_query", "retrieve")
workflow.add_edge("retrieve", "validate_context")
workflow.add_conditional_edges(
    "validate_context",
    route_after_validation,
    {
        "grade_documents": "grade_documents",
        "handle_degraded_context": "handle_degraded_context",
    },
)
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate",
        "handle_irrelevant": "handle_irrelevant",
    },
)
workflow.add_edge("generate", END)
workflow.add_edge("handle_irrelevant", END)
workflow.add_edge("handle_degraded_context", END)

# Compile
rag_app = workflow.compile()
