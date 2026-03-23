from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import traceback
from graph_mit import agent_app 

app = FastAPI(title="Meeting Intelligence API", version="1.0.0")

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_text(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        result = agent_app.invoke({"original_text": request.text})
        
        return {
            "title": result.get("title", "Title not found"),
            "summary": result.get("summary", "Summary not in context"),
            "tasks": result.get("tasks", []),
            "risks": result.get("risks", []), # <-- Added Risks
            "decision_points": result.get("decision_points", [])
        }
    except Exception as e:
        print("--- ERROR TRACEBACK ---")
        traceback.print_exc() 
        print("-----------------------")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)