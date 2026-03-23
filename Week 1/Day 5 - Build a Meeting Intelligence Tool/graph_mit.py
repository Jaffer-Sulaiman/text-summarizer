import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from schema_mit import AgentState, ExtractionResult
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", # Changed to standard 1.5-pro for stability
    temperature=0.2, 
    max_tokens=2048
)

def master_extraction_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI Scrum Master and Meeting Analyst. Analyze the provided meeting transcript and extract a title, summary, tasks, risks, and decision points.

        Use the following adaptable definitions:

        1. TASKS:
        - Definition: Concrete action items, assigned duties, or next steps.
        - Look for: Assigned duties with deadlines (e.g., "John will review the contract"), or imperative commands.

        2. RISKS:
        - Definition: Potential blockers, vulnerabilities, delays, budget concerns, or unresolved issues discussed by the team.
        - Look for: Words indicating concern (e.g., "might delay", "we are blocked by", "I'm worried about", "dependency").

        3. DECISION POINTS:
        - Definition: Finalized agreements, approvals, strategic pivots, or established facts.
        - Look for: Formal consensus (e.g., "The board approved", "We decided to").

        CRITICAL CONSTRAINTS:
        - Grounding: Base your extraction STRICTLY on the provided text. Do not invent details.
        - Empty States: If there are absolutely no tasks, risks, or decisions, you MUST return an empty list [] for those fields.
        - Insufficient Input: If the input text is too short, return "Summary not in context" for the summary, "Title not found" for the title, and empty lists for the rest.
        """),
        ("user", "{text}")
    ])
    
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = prompt | structured_llm
    
    response = chain.invoke({"text": state["original_text"]})
    
    if response:
        return {
            "title": response.title,
            "summary": response.summary,
            "tasks": response.tasks,
            "risks": response.risks, # <-- Added Risks
            "decision_points": response.decision_points
        }
    else:
        return {"title": "Error", "summary": "Failed to extract data.", "tasks": [], "risks": [], "decision_points": []}

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("extractor", master_extraction_node)
    builder.add_edge(START, "extractor")
    builder.add_edge("extractor", END)
    return builder.compile()

agent_app = build_graph()