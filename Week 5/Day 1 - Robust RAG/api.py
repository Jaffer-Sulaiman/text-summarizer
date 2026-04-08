from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import traceback
import pypdf
import io

from vectorstore import vector_manager
from graph import rag_app
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(
    title="Robust RAG API",
    description="A production-ready Retrieval Augmented Generation API capable of document ingestion and conversational Q&A.",
    version="1.0.0"
)

# Enable CORS for frontend interoperability
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================
class MessageBody(BaseModel):
    role: str = Field(description="Role of the sender ('user' or 'ai')")
    content: str = Field(description="Content of the message")

class ChatRequest(BaseModel):
    query: str = Field(..., max_length=1500, description="The user's question, limited to prevent buffer overflow.")
    history: List[MessageBody] = Field(default_factory=list, description="Array of past messages for memory context.")

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Ingest a PDF or TXT file, chunk it, and store into ChromaDB."""
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    try:
        contents = await file.read()
        extracted_text = ""

        # Extract Text
        if filename.endswith('.txt'):
            extracted_text = contents.decode('utf-8')
        elif filename.endswith('.pdf'):
            reader = pypdf.PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found. Scanned PDFs are not currently supported.")

        # Ingest to VectorStore
        result = vector_manager.ingest_document(extracted_text, file.filename)
        
        if result["status"] == "error":
             raise HTTPException(status_code=500, detail=result["message"])

        return result

    except Exception as e:
        print("--- UPLOAD ERROR ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/chat")
async def chat(request: ChatRequest):
    """Answer a user's question using RAG and conversational history."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    # Guardrail 1: Check if DB is empty
    if vector_manager.is_empty():
        return {"answer": "My knowledge base is empty! Please upload a document first before asking questions."}

    try:
        # Reconstruct LangChain memory messages
        formatted_messages = []
        for msg in request.history:
            if msg.role == "user" or msg.role == "human":
                formatted_messages.append(HumanMessage(content=msg.content))
            else:
                formatted_messages.append(AIMessage(content=msg.content))

        # We append the current query to the state natively via the graph logic instead
        # The graph inputs are 'question' and 'messages'
        inputs = {
            "question": request.query,
            "messages": formatted_messages
        }

        # Invoke the robust LangGraph orchestrator
        final_state = rag_app.invoke(inputs)

        # Return synthesized answer
        answer = final_state.get("answer", "An unexpected error occurred during logic routing.")
        
        return {"answer": answer}

    except Exception as e:
        print("--- CHAT ERROR ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM/Graph execution failed: {str(e)}")

# For standalone execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
