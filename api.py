# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback

# Import the compiled graph from the previous step
# (Assuming you saved the Step 3 code in a file named `graph.py`)
from graph import agent_app 

app = FastAPI(
    title="Executive Text Analyzer API",
    description="API for summarizing text and extracting actions/decisions using Gemini and LangGraph.",
    version="1.0.0"
)

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        # Invoke the LangGraph workflow
        # LangGraph handles the parallel execution of our 3 nodes under the hood
        result = agent_app.invoke({"original_text": request.text})
        
        # Return a clean JSON response stripping out the original text to save bandwidth
        return {
            "title": result.get("title", "Untitled Document"), # <-- Add this
            "summary": result.get("summary", ""),
            "action_items": result.get("action_items", []),
            "key_decisions": result.get("key_decisions", [])
        }
    except Exception as e:
        # Catch and surface any LLM or Graph execution error
        # Print the full error to the backend terminal
        print("--- ERROR TRACEBACK ---")
        traceback.print_exc() 
        print("-----------------------")
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)