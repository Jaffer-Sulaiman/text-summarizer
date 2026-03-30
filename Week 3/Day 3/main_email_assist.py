from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

# Import the compiled Map-Reduce agent from agent.py
from agent_email_assist import email_agent

# Set up logging to track the parallel execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Email Processing Agent API - Map Reduce", version="2.0")

# --- Pydantic Models for API Validation ---
class EmailRequest(BaseModel):
    raw_email: str

class EmailResponse(BaseModel):
    is_valid: bool
    master_summary: str | None = None
    urgency_level: str | None = None
    reply_suggestion: str | None = None
    error: str | None = None

# --- API Endpoints ---
@app.post("/api/process-email", response_model=EmailResponse)
async def process_email(request: EmailRequest):
    logger.info("Received new email. Starting Map-Reduce pipeline.")
    
    # Initialize the LangGraph state
    initial_state = {"raw_email": request.raw_email}
    
    try:
        # Invoke the compiled LangGraph agent
        final_state = email_agent.invoke(initial_state)
        
        # Guardrail check: Did the agent reject the email early?
        if not final_state.get("is_valid"):
            logger.warning("Email was rejected by the validation node.")
            return EmailResponse(
                is_valid=False,
                error="Email payload is missing or too short to process."
            )
        
        logger.info(f"Processing complete. Urgency identified: {final_state.get('urgency_level')}")
        
        # Return the extracted state variables back to the UI
        return EmailResponse(
            is_valid=True,
            master_summary=final_state.get("master_summary"),
            urgency_level=final_state.get("urgency_level"),
            reply_suggestion=final_state.get("reply_suggestion"),
            error=None
        )
        
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent encountered an error: {str(e)}")

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)