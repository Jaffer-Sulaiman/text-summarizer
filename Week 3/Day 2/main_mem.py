from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uvicorn

from agent_mem import email_agent

app = FastAPI(title="Email Copilot API")

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Pass the user's message into the state
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    # Configure the thread ID for the checkpointer
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        # Run the agent
        final_state = email_agent.invoke(inputs, config=config)
        
        # Extract the very last message the AI generated to send back to the UI
        ai_response = final_state["messages"][-1].content
        return ChatResponse(response=ai_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)