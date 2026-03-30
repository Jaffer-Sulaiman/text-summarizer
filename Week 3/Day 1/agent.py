import os
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Ensure your GOOGLE_API_KEY is set in your environment
# os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

# ---------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------
class EmailState(TypedDict):
    raw_email: str
    is_valid: bool
    error: Optional[str]
    summary: Optional[str]
    intent: Optional[str]
    draft: Optional[str]

# Initialize the Gemini Model (1.5 Flash is incredibly fast and cost-effective for these rapid steps)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.2)

# ---------------------------------------------------------
# 2. Node Functions (Strict Separation of Steps)
# ---------------------------------------------------------
def check_email(state: EmailState):
    """Validates the incoming email payload."""
    email_content = state.get("raw_email", "").strip()
    
    # Guardrail: Check for empty or impossibly short emails
    if not email_content or len(email_content) < 15:
        return {"is_valid": False, "error": "Email payload is missing or too short to process."}
    
    return {"is_valid": True, "error": None}

def summarize_email(state: EmailState):
    """Extracts the core message and key entities."""
    prompt = f"Summarize the following email in 2-3 concise sentences. Highlight key entities, dates, and any sense of urgency:\n\n{state['raw_email']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

# Guardrail: Force the LLM to output ONLY one of these exact strings
class IntentClassification(BaseModel):
    intent: Literal["support_request", "sales_inquiry", "meeting_invite", "spam", "other"] = Field(
        description="The strictly classified intent of the email."
    )

def identify_intent(state: EmailState):
    """Classifies the email using structured JSON output."""
    structured_llm = llm.with_structured_output(IntentClassification)
    prompt = f"Analyze this email summary and classify its intent into one of the predefined categories:\n\n{state['summary']}"
    
    response = structured_llm.invoke(prompt)
    return {"intent": response.intent}

def draft_response(state: EmailState):
    """Drafts a context-aware reply based on intent and summary."""
    system_prompt = (
        f"You are a professional corporate assistant. Draft a polite, concise reply to the user's email. "
        f"The identified intent of their email is: '{state['intent']}'. "
        f"Use the summary for context: '{state['summary']}'. Do not invent facts."
    )
    user_prompt = f"Original email:\n{state['raw_email']}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {"draft": response.content}

# ---------------------------------------------------------
# 3. Conditional Routing Logic
# ---------------------------------------------------------
def route_after_check(state: EmailState):
    """Stop processing if the email is invalid."""
    if not state.get("is_valid"):
        return "end"
    return "summarize"

def route_after_intent(state: EmailState):
    """Do not waste compute drafting replies to spam or irrelevant emails."""
    if state.get("intent") in ["spam", "other"]:
        return "end"
    return "draft"

# ---------------------------------------------------------
# 4. Graph Compilation
# ---------------------------------------------------------
builder = StateGraph(EmailState)

# Add Nodes
builder.add_node("check", check_email)
builder.add_node("summarize", summarize_email)
builder.add_node("identify", identify_intent)
builder.add_node("draft", draft_response)

# Define Flow
builder.set_entry_point("check")

builder.add_conditional_edges(
    "check",
    route_after_check,
    {"summarize": "summarize", "end": END}
)
builder.add_edge("summarize", "identify")

builder.add_conditional_edges(
    "identify",
    route_after_intent,
    {"draft": "draft", "end": END}
)
builder.add_edge("draft", END)

# Export the compiled agent
email_agent = builder.compile()