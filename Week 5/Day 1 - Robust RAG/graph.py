from typing import Annotated, List, TypedDict, Dict, Any
from typing_extensions import NotRequired
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Import our custom vector store manager
from vectorstore import vector_manager

# ==============================================================================
# STATE DEFINITION
# ==============================================================================
class AgentState(TypedDict):
    """The State of the RAG graph."""
    messages: Annotated[List[BaseMessage], add_messages] # Conversational memory
    question: str
    documents: List[str]
    is_relevant: str
    answer: str

# ==============================================================================
# LLM CONFIGURATION
# ==============================================================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.2, # Low temperature for more factual, grounded responses
    convert_system_message_to_human=True # Required for some older runnables, safe to keep
)

class GradeOutput(BaseModel):
    """Pydantic model for structured grading outputs."""
    score: str = Field(description="Relevance score: 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeOutput)

# ==============================================================================
# NODES
# ==============================================================================
def retrieve(state: AgentState):
    """Retrieve documents from ChromaDB."""
    print("---RETRIEVE---")
    question = state["question"]
    # Retrieve top 4 chunks
    docs = vector_manager.retrieve(question, top_k=4)
    # Extract string content from Document objects
    doc_strings = [doc.page_content for doc in docs]
    return {"documents": doc_strings}

def grade_documents(state: AgentState):
    """Check if the retrieved documents are relevant to the question."""
    print("---CHECK RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"is_relevant": "no"}

    context = "\n\n".join(documents)
    
    # Prompt for grading
    prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.
    Here is the retrieved document context: \n\n {context} \n\n
    Here is the user question: {question}
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    try:
        result = structured_llm_grader.invoke(prompt)
        score = result.score.lower().strip()
    except Exception as e:
        print(f"Error in grader: {e}")
        score = "yes" # Fallback to generation if grading fails
        
    print(f"Relevance Score: {score}")
    return {"is_relevant": score}

def generate(state: AgentState):
    """Generate answer using RAG and conversational history."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    messages = state.get("messages", [])
    
    context = "\n\n".join(documents)
    
    # We construct a system prompt that enforces strict grounding
    system_prompt = f"""You are a helpful and polite professional AI assistant. 
    You are answering questions based ONly on the provided Context.
    
    Context:
    {context}
    
    Rules:
    1. Answer the question using ONLY the context above.
    2. Do NOT hallucinate information or rely on outside knowledge.
    3. If the answer is not contained in the Context, politely state that you do not know based on the provided documents.
    4. Provide clear, well-structured, and concise answers."""

    # We build the chat history
    formatted_messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Append past history (excluding the current question)
    # We map BaseMessage roles to simple string roles
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
             formatted_messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
             formatted_messages.append(("ai", msg.content))
             
    # Append the current question
    formatted_messages.append(("human", question))
    
    response = llm.invoke(formatted_messages)
    return {"answer": response.content}

def handle_irrelevant(state: AgentState):
    """Handle cases where retrieved docs do not answer the question."""
    print("---IRRELEVANT / OUT OF SCOPE---")
    return {"answer": "I'm sorry, I cannot answer this question because it is outside the scope of the documents you've uploaded. Please ask me something related to the knowledge base!"}

# ==============================================================================
# CONDITIONAL EDGES
# ==============================================================================
def decide_to_generate(state: AgentState):
    """Determine whether to generate an answer, or fallback to irrelavant."""
    print("---DECIDE TO GENERATE---")
    is_relevant = state["is_relevant"]
    
    if is_relevant == "yes":
        return "generate"
    else:
        return "handle_irrelevant"

# ==============================================================================
# GRAPH COMPILATION
# ==============================================================================
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("handle_irrelevant", handle_irrelevant)

# Flow
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "handle_irrelevant": "handle_irrelevant",
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("handle_irrelevant", END)

# Compile
rag_app = workflow.compile()
