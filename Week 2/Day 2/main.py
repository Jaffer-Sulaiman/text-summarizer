import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import our compiled LangGraph application
from agent import app as agent_app

app = FastAPI(title="Tool Calling Agent API")

# --- Pydantic Schemas ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Hardcoded thread_id for our single-user demo memory configuration
    config = {"configurable": {"thread_id": "demo_user"}}
    
    try:
        # We use ainvoke() because our tool functions (like weather/currency) are async
        # 'add_messages' in our graph state handles appending this safely to the history
        result = await agent_app.ainvoke(
            {"messages": [("user", request.message)]}, 
            config=config
        )
        
        # The graph returns the full state. The last message is the AI's final response.
        final_message = result["messages"][-1].content
        return ChatResponse(response=final_message)
        
    except Exception as e:
        # Catch unexpected backend failures
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)