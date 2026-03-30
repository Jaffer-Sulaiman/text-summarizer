from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

# Import the compiled agent from our previous file (assuming it's named agent.py)
from agent import email_agent

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Email Processing Agent API", version="1.0")

# --- Pydantic Models for API Validation ---
class EmailRequest(BaseModel):
    raw_email: str

class EmailResponse(BaseModel):
    is_valid: bool
    summary: str | None = None
    intent: str | None = None
    draft: str | None = None
    error: str | None = None

# --- API Endpoints ---
@app.post("/api/process-email", response_model=EmailResponse)
async def process_email(request: EmailRequest):
    logger.info("Received new email for processing.")
    
    # Initialize the LangGraph state
    initial_state = {"raw_email": request.raw_email}
    
    try:
        # Invoke the compiled LangGraph agent
        # .invoke() runs the graph synchronously; use .ainvoke() if your nodes were async
        final_state = email_agent.invoke(initial_state)
        
        logger.info(f"Processing complete. Intent identified: {final_state.get('intent')}")
        
        return EmailResponse(
            is_valid=final_state.get("is_valid", False),
            summary=final_state.get("summary"),
            intent=final_state.get("intent"),
            draft=final_state.get("draft"),
            error=final_state.get("error")
        )
        
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent encountered an error: {str(e)}")

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)