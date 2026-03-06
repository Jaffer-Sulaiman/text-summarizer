from typing import List, TypedDict
from pydantic import BaseModel, Field

# 1. The Master Pydantic Model (Hardened)
class ExtractionResult(BaseModel):
    title: str = Field(
        default="Untitled Document",
        description="A short, professional title. If the text is too short or lacks context, return 'Untitled Document'."
    )
    summary: str = Field(
        default="Insufficient text provided.",
        description="An executive summary. If the text is too short to summarize, return 'Insufficient text provided for a summary.'"
    )
    # Using default_factory=list prevents mutable default bugs in Python
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