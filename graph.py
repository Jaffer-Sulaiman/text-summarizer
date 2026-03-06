import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from schema import AgentState, ActionItems, KeyDecisions
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the .env file into os.environ
load_dotenv()

# Initialize Gemini (1.5 Pro is ideal for long context)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview", 
    temperature=0.2, # Low temperature for more factual extraction
    max_tokens=2048
)

# --- Define the Nodes ---

def summarize_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert executive assistant. Provide a concise, highly readable summary of the following text."),
        ("user", "{text}")
    ])
    
    # We remove StrOutputParser() and just use the base chain
    chain = prompt | llm 
    response = chain.invoke({"text": state["original_text"]})
    
    content = response.content
    
    # Bulletproof extraction: 
    # If Gemini returns the raw list/dict payload, we drill down to grab just the 'text'
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        summary_text = content[0].get("text", "")
    else:
        # Fallback if it actually does return a string
        summary_text = str(content) 
        
    return {"summary": summary_text}

def extract_actions_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract all action items, task assignments, and next steps from the text. Return them as a structured list."),
        ("user", "{text}")
    ])
    # Force Gemini to output JSON matching our Pydantic schema
    structured_llm = llm.with_structured_output(ActionItems)
    chain = prompt | structured_llm
    response = chain.invoke({"text": state["original_text"]})
    return {"action_items": response.items if response else []}

def extract_decisions_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract all key decisions, finalized agreements, or policy approvals from the text. Return them as a structured list."),
        ("user", "{text}")
    ])
    structured_llm = llm.with_structured_output(KeyDecisions)
    chain = prompt | structured_llm
    response = chain.invoke({"text": state["original_text"]})
    return {"key_decisions": response.decisions if response else []}

def generate_title_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert editor. Generate a short, professional title (maximum 6 words) for the following text. Do not use quotes. Just output the title."),
        ("user", "{text}")
    ])
    
    chain = prompt | llm 
    response = chain.invoke({"text": state["original_text"]})
    
    content = response.content
    # Using our bulletproof extraction just to be safe
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        title_text = content[0].get("text", "").strip()
    else:
        title_text = str(content).strip()
        
    return {"title": title_text}


# --- Build the Graph ---
def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("title_generator", generate_title_node) # <-- Add this
    builder.add_node("summarizer", summarize_node)
    builder.add_node("action_extractor", extract_actions_node)
    builder.add_node("decision_extractor", extract_decisions_node)

    # Wire all 4 to start at the same time
    builder.add_edge(START, "title_generator") # <-- Add this
    builder.add_edge(START, "summarizer")
    builder.add_edge(START, "action_extractor")
    builder.add_edge(START, "decision_extractor")

    builder.add_edge("title_generator", END) # <-- Add this
    builder.add_edge("summarizer", END)
    builder.add_edge("action_extractor", END)
    builder.add_edge("decision_extractor", END)

    return builder.compile()

# Instantiate the compiled graph
agent_app = build_graph()