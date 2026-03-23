from typing import List, TypedDict
from pydantic import BaseModel, Field

# 1. The Master Pydantic Model (Hardened)
class ExtractionResult(BaseModel):
    title: str = Field(
        default="Title not found", # <-- Updated
        description="A short, professional title. If the text is too short or lacks context, return 'Title not found'."
    )
    summary: str = Field(
        default="Summary not in context", # <-- Updated
        description="An executive summary. If the text is too short to summarize, return 'Summary not in context.'"
    )
    action_items: List[str] = Field(
        default_factory=list, 
        description="A list of clearly defined action items. If absolutely no action items are found, return an empty list []."
    )
    key_decisions: List[str] = Field(
        default_factory=list, 
        description="A list of final decisions made. If absolutely no decisions are found, return an empty list []."
    )

# 2. LangGraph State
class AgentState(TypedDict):
    original_text: str
    title: str 
    summary: str
    action_items: List[str]
    key_decisions: List[str]