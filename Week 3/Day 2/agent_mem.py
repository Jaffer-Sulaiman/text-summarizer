import os
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.2)

# ---------------------------------------------------------
# 1. State Definition (Now with Memory)
# ---------------------------------------------------------
class EmailState(TypedDict):
    # LangGraph automatically appends new messages to this list
    messages: Annotated[list[BaseMessage], add_messages]
    raw_email: str | None
    is_valid: bool
    summary: str | None
    intent: str | None
    draft: str | None

# ---------------------------------------------------------
# 2. Node Functions
# ---------------------------------------------------------
def check_email(state: EmailState):
    """Reads the first user message as the raw email."""
    latest_msg = state["messages"][-1].content.strip()
    
    if not latest_msg or len(latest_msg) < 15:
        error_msg = "❌ This email is too short or invalid. Please paste a full email."
        return {"is_valid": False, "messages": [AIMessage(content=error_msg)]}
    
    return {"is_valid": True, "raw_email": latest_msg}

def summarize_email(state: EmailState):
    prompt = f"Summarize the following email in 2-3 concise sentences. Highlight key entities:\n\n{state['raw_email']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"summary": response.content}

class IntentClassification(BaseModel):
    intent: Literal["support_request", "sales_inquiry", "meeting_invite", "spam", "other"]

def identify_intent(state: EmailState):
    structured_llm = llm.with_structured_output(IntentClassification)
    prompt = f"Analyze this email summary and classify its intent:\n\n{state['summary']}"
    response = structured_llm.invoke(prompt)
    return {"intent": response.intent}

def draft_response(state: EmailState):
    """Creates the initial draft and outputs it to the chat."""
    system_prompt = (
        f"You are a professional corporate assistant. Draft a polite reply to the user's email. "
        f"Intent: '{state['intent']}'. Summary: '{state['summary']}'."
    )
    user_prompt = f"Original email:\n{state['raw_email']}"
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    draft_text = response.content
    
    # Format the AI's chat response
    chat_reply = f"✅ **Intent:** {state['intent'].replace('_', ' ').title()}\n\n**Draft V1:**\n{draft_text}\n\n*How would you like to tweak this? (e.g., 'Make it friendlier', 'Make it shorter')*"
    return {"draft": draft_text, "messages": [AIMessage(content=chat_reply)]}

def refine_draft(state: EmailState):
    """Applies user feedback to the existing draft."""
    feedback = state["messages"][-1].content
    
    system_prompt = (
        f"You are an assistant refining an email draft based on user feedback.\n"
        f"Original Email: {state['raw_email']}\n"
        f"Current Draft: {state['draft']}\n\n"
        f"Apply the user's feedback to rewrite the draft."
    )
    
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=feedback)])
    new_draft = response.content
    
    chat_reply = f"**Updated Draft:**\n{new_draft}\n\n*Any other changes?*"
    return {"draft": new_draft, "messages": [AIMessage(content=chat_reply)]}

# ---------------------------------------------------------
# 3. Routing Logic
# ---------------------------------------------------------
def route_initial(state: EmailState):
    """Determines if the user is submitting a new email or giving feedback."""
    if state.get("draft"):
        return "refine"
    return "check"

def route_after_check(state: EmailState):
    if not state.get("is_valid"):
        return END
    return "summarize"

def route_after_intent(state: EmailState):
    if state.get("intent") in ["spam", "other"]:
        return END
    return "draft"

# ---------------------------------------------------------
# 4. Graph Compilation with Checkpointer
# ---------------------------------------------------------
builder = StateGraph(EmailState)

builder.add_node("check", check_email)
builder.add_node("summarize", summarize_email)
builder.add_node("identify", identify_intent)
builder.add_node("draft", draft_response)
builder.add_node("refine", refine_draft)

# The router decides where the message goes
builder.add_conditional_edges(START, route_initial, {"check": "check", "refine": "refine"})

builder.add_conditional_edges("check", route_after_check, {"summarize": "summarize", END: END})
builder.add_edge("summarize", "identify")
builder.add_conditional_edges("identify", route_after_intent, {"draft": "draft", END: END})

builder.add_edge("draft", END)
builder.add_edge("refine", END)

# Compile with In-Memory Checkpointer
memory = MemorySaver()
email_agent = builder.compile(checkpointer=memory)