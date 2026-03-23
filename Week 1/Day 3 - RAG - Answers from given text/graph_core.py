from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Assuming 'llm' and 'vectorstore' are imported from our Phase 1/2 code
from rag_core import llm, vectorstore   

# ==========================================
# 1. State & Pydantic Definitions
# ==========================================

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    transcript_id: str
    context: List[Document]
    relevance_score: str  # "yes" or "no"
    answer: str
    sources: List[str]

class GraderOutput(BaseModel):
    """Binary score for relevance check."""
    score: str = Field(
        description="Are the retrieved documents relevant to the question? Answer 'yes' or 'no'"
    )

class GenerativeOutput(BaseModel):
    """Structured response to ensure we always get sources."""
    answer: str = Field(
        description="The detailed answer to the user's question based strictly on the context."
    )
    sources: List[str] = Field(
        description="List of exact quotes or snippets from the context used to formulate the answer."
    )

# ==========================================
# 2. Node Functions
# ==========================================

def retrieve(state: GraphState):
    """Retrieves documents based on the selected transcript_id."""
    print("---RETRIEVING CONTEXT---")
    question = state["question"]
    transcript_id = state["transcript_id"]

    # We use a metadata filter to ONLY search the selected transcript
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5, 
            "filter": {"doc_id": transcript_id}
        }
    )
    documents = retriever.invoke(question)
    return {"context": documents}

def grade_documents(state: GraphState):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---GRADING DOCUMENTS---")
    question = state["question"]
    documents = state["context"]

    # Bind Pydantic model for structured output
    structured_llm_grader = llm.with_structured_output(GraderOutput)

    system = """You are a strict grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as 'yes'. \n
    If the document has absolutely nothing to do with the question, grade it as 'no'."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    # Let's combine document content for grading
    doc_text = "\n\n".join([doc.page_content for doc in documents])
    
    chain = grade_prompt | structured_llm_grader
    result = chain.invoke({"document": doc_text, "question": question})
    
    return {"relevance_score": result.score.lower()}

def generate_answer(state: GraphState):
    """Generates an answer strictly from the context."""
    print("---GENERATING ANSWER---")
    question = state["question"]
    documents = state["context"]

    structured_llm_generator = llm.with_structured_output(GenerativeOutput)

    system = """You are a helpful assistant answering questions based ONLY on the provided meeting transcript context.
    Do not use outside knowledge. If the answer is not in the context, do not answer.
    Provide a clear, concise answer and list the exact snippets you used as sources."""
    
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Context: \n\n {context} \n\n User Question: {question}"),
    ])

    doc_text = "\n\n".join([doc.page_content for doc in documents])
    
    chain = generate_prompt | structured_llm_generator
    result = chain.invoke({"context": doc_text, "question": question})
    
    return {"answer": result.answer, "sources": result.sources}

def polite_refusal(state: GraphState):
    """Handles out-of-context questions."""
    print("---POLITE REFUSAL---")
    return {
        "answer": "I'm sorry, but I cannot find the answer to that question in the selected transcript.",
        "sources": []
    }

# ==========================================
# 3. Routing Functions
# ==========================================

def route_question(state: GraphState) -> Literal["generate_answer", "polite_refusal"]:
    """Routes to generation or refusal based on the grader's score."""
    if state["relevance_score"] == "yes":
        return "generate_answer"
    return "polite_refusal"

# ==========================================
# 4. Graph Compilation
# ==========================================

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("polite_refusal", polite_refusal)

# Define Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge from Grader
workflow.add_conditional_edges(
    "grade_documents",
    route_question,
)

# Connect generators to END
workflow.add_edge("generate_answer", END)
workflow.add_edge("polite_refusal", END)

# Compile the final graph
app_graph = workflow.compile()