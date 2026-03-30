import os
import operator
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send

# Initialize Gemini 1.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.2)

# --- 1. State Definitions ---

class EmailState(TypedDict):
    raw_email: str
    is_valid: bool
    chunks: list[str]
    # operator.add tells LangGraph to append parallel outputs into a single list
    chunk_summaries: Annotated[list[str], operator.add] 
    master_summary: str | None
    urgency_level: str | None
    reply_suggestion: str | None

# We define a minimal sub-state for the parallel nodes
class ChunkState(TypedDict):
    chunk_text: str


# --- 2. Node Functions ---

def check_and_chunk(state: EmailState):
    """Validates the email and splits it into chunks if it's too long."""
    raw = state.get("raw_email", "").strip()
    
    if not raw or len(raw) < 15:
         return {"is_valid": False}
         
    # Simple chunking logic (e.g., ~4000 characters per chunk)
    chunk_size = 4000
    chunks = [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
    
    return {"is_valid": True, "chunks": chunks, "chunk_summaries": []}

def summarize_chunk(state: ChunkState):
    """The Map Node: Summarizes a single chunk. Runs in parallel."""
    prompt = f"Summarize the key points of this email excerpt:\n\n{state['chunk_text']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Return as a list so operator.add can combine them in the main state
    return {"chunk_summaries": [response.content]}

def reduce_summaries(state: EmailState):
    """The Reduce Node: Combines all chunk summaries into a master summary."""
    # If there was only 1 chunk, we just pass it through as the master summary
    if len(state["chunk_summaries"]) == 1:
        return {"master_summary": state["chunk_summaries"][0]}
        
    combined_text = "\n\n".join(state["chunk_summaries"])
    prompt = f"Combine the following excerpt summaries into one cohesive master summary of the entire email thread:\n\n{combined_text}"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"master_summary": response.content}

# --- Strict Urgency Classification ---
class UrgencyClassification(BaseModel):
    urgency: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="The strictly classified urgency level of the email."
    )

def detect_urgency(state: EmailState):
    """Evaluates the master summary to determine urgency."""
    structured_llm = llm.with_structured_output(UrgencyClassification)
    prompt = f"Analyze this email summary and classify its urgency strictly as Low, Medium, High, or Critical:\n\n{state['master_summary']}"
    
    response = structured_llm.invoke(prompt)
    return {"urgency_level": response.urgency}

def draft_reply(state: EmailState):
    """Drafts the final response based on urgency and the master summary."""
    system_prompt = (
        f"You are a professional assistant. Draft a reply to the email. "
        f"The urgency level is: '{state['urgency_level']}'. "
        f"If the urgency is High or Critical, ensure the tone is highly responsive and prioritizing. "
        f"Use this summary for context: '{state['master_summary']}'."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original context:\n{state['raw_email'][:2000]}...") # Pass a snippet for tone matching
    ])
    return {"reply_suggestion": response.content}


# --- 3. Dynamic Routing Logic ---

def route_after_check(state: EmailState):
    """If invalid, end. If valid, trigger parallel summarization."""
    if not state.get("is_valid"):
        return END
        
    # The `Send` API tells LangGraph to spin up a "summarize_chunk" node 
    # for EVERY item in the chunks array, passing it the ChunkState.
    return [Send("summarize_chunk", {"chunk_text": chunk}) for chunk in state["chunks"]]

# --- 4. Graph Compilation ---
builder = StateGraph(EmailState)

# Add Nodes
builder.add_node("check", check_and_chunk)
builder.add_node("summarize_chunk", summarize_chunk)
builder.add_node("reduce", reduce_summaries)
builder.add_node("detect_urgency", detect_urgency)
builder.add_node("draft", draft_reply)

# Define Flow
builder.set_entry_point("check")

# Fan-out: Check node maps to N summarize_chunk nodes
builder.add_conditional_edges("check", route_after_check, ["summarize_chunk", END])

# Fan-in: All summarize_chunk nodes converge on the reduce node
builder.add_edge("summarize_chunk", "reduce")

# Standard linear flow from there
builder.add_edge("reduce", "detect_urgency")
builder.add_edge("detect_urgency", "draft")
builder.add_edge("draft", END)

email_agent = builder.compile()