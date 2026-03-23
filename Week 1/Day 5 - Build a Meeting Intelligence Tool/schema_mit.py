from typing import List, TypedDict
from pydantic import BaseModel, Field

# 1. The Master Pydantic Model (Hardened for Meetings)
class ExtractionResult(BaseModel):
    title: str = Field(
        default="Title not found",
        description="A short, professional title for the meeting. If the text lacks context, return 'Title not found'."
    )
    summary: str = Field(
        default="Summary not in context",
        description="An executive summary of the meeting. If the text is too short, return 'Summary not in context.'"
    )
    tasks: List[str] = Field(
        default_factory=list, 
        description="A list of assigned tasks and next steps. If none are found, return an empty list []."
    )
    risks: List[str] = Field(
        default_factory=list, 
        description="A list of potential blockers, vulnerabilities, or risks discussed. If none, return an empty list []."
    )
    decision_points: List[str] = Field(
        default_factory=list, 
        description="A list of final decisions or agreements. If none are found, return an empty list []."
    )

# 2. LangGraph State
class AgentState(TypedDict):
    original_text: str
    title: str 
    summary: str
    tasks: List[str]
    risks: List[str]
    decision_points: List[str]