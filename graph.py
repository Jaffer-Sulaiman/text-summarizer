import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from schema import AgentState, ExtractionResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini 
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview", 
    temperature=0.2, 
    max_tokens=2048
)

# --- Define the Single Master Node ---
def master_extraction_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI reading assistant and document analyst. Your task is to analyze the provided text and extract a title, an executive summary, action items, and key decisions.

        The input text may range from corporate meeting transcripts to general articles, blogs, or research papers. Use the following adaptable definitions:

        1. ACTION ITEMS:
        - Definition: Tasks, concrete next steps, actionable advice, recommendations, or guidelines presented in the text.
        - Look for: Assigned duties with deadlines (e.g., "John will review the contract"), imperative commands, or practical tips for the reader (e.g., "Eat more vegetables," "Always double-check your work").

        2. KEY DECISIONS (OR CONCLUSIONS):
        - Definition: Finalized agreements, established facts, primary recommendations, or main takeaways from the text.
        - Look for: Formal consensus (e.g., "The board approved the budget"), definitive statements, core arguments, or the primary conclusions reached by the author/speakers.

        CRITICAL CONSTRAINTS:
        - Grounding: Base your extraction STRICTLY on the provided text. Do not invent, infer, or hallucinate details.
        - Empty States: If the text is purely narrative and contains absolutely no actionable advice, tasks, or meaningful conclusions, you MUST return an empty list [] for those fields.
        - Insufficient Input: If the input text is too short or lacks meaningful context, state "Insufficient text provided for meaningful analysis." in the summary and return empty lists for actions and decisions.
        """),
        ("user", "{text}")
    ])
    
    # Force Gemini to output the massive JSON structure
    structured_llm = llm.with_structured_output(ExtractionResult)
    chain = prompt | structured_llm
    
    # Execute the single API call
    response = chain.invoke({"text": state["original_text"]})
    
    # Map the resulting Pydantic object back to our AgentState dictionary
    if response:
        return {
            "title": response.title,
            "summary": response.summary,
            "action_items": response.action_items,
            "key_decisions": response.key_decisions
        }
    else:
        # Fallback in case of a highly unusual parsing failure
        return {"title": "Error", "summary": "Failed to extract data.", "action_items": [], "key_decisions": []}

# --- Build the Graph ---
def build_graph():
    builder = StateGraph(AgentState)

    # Add our single node
    builder.add_node("extractor", master_extraction_node)

    # Wire the linear graph
    builder.add_edge(START, "extractor")
    builder.add_edge("extractor", END)

    return builder.compile()

# Instantiate the compiled graph
agent_app = build_graph()