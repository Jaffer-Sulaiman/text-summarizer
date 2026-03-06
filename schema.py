from typing import List, TypedDict
from pydantic import BaseModel, Field

# 1. Pydantic Models for Structured LLM Output
class ActionItems(BaseModel):
    items: List[str] = Field(description="A list of clearly defined action items, tasks, or next steps extracted from the text.")

class KeyDecisions(BaseModel):
    decisions: List[str] = Field(description="A list of final decisions, agreements, or conclusions made in the text.")

# 2. LangGraph State
class AgentState(TypedDict):
    original_text: str
    title: str # <-- Add this line
    summary: str
    action_items: List[str]
    key_decisions: List[str]