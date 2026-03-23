from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
# 1. Changed import from OpenAI to Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import the tools list from our previous step
from tools import tools 

# --- 1. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 2. Setup the LLM ---
# 2. Replaced ChatOpenAI with ChatGoogleGenerativeAI.
# gemini-1.5-flash is excellent and fast for utility tool calling. 
# (Make sure you have GOOGLE_API_KEY set in your environment variables)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) 
llm_with_tools = llm.bind_tools(tools)

# --- 3. The Guardrail System Prompt ---
SYSTEM_PROMPT = """You are a highly capable, specialized utility assistant.
You have access to three specific tools:
1. get_weather: For current weather data.
2. currency_converter: For live exchange rates.
3. web_search: For fetching current facts or news.

CRITICAL INSTRUCTIONS:
- If a user asks a general, conversational, or out-of-scope question (e.g., "Hi", "Write a poem", "What is the meaning of life?"), DO NOT attempt to use any tools. 
- Instead, provide a brief, polite response (maximum 2 sentences) and guide them back to your core capabilities. Example: "Hello! I am a specialized assistant here to help with live weather, currency conversions, and web searches. What would you like to look up today?"
- If a user's tool request is missing required parameters (like a city for the weather), ask them for the missing information directly. Do not guess parameters.
- If a tool returns an error message, apologize gracefully to the user and explain that the specific service is currently unavailable based on the error provided.
"""

# --- 4. Define the Nodes ---
def agent_node(state: AgentState):
    """The main reasoning engine."""
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# --- 5. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# --- 6. Compile with Memory ---
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)