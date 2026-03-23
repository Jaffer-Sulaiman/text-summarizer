import gradio as gr
import httpx

# The URL where our FastAPI server is running
API_URL = "http://127.0.0.1:8000/chat"

def chat_with_agent(message, history):
    """Sends the user message to the FastAPI backend and returns the text response."""
    try:
        # Using httpx to call our API gateway
        with httpx.Client(timeout=30.0) as client:
            response = client.post(API_URL, json={"message": message})
            response.raise_for_status()
            
            # Parse the FastAPI ChatResponse schema
            data = response.json()
            return data["response"]
            
    except httpx.ReadTimeout:
        return "System Error: The agent took too long to respond."
    except Exception as e:
        return f"System Error: Could not communicate with the backend API. ({str(e)})"

# --- Build the Gradio UI ---
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="Fundamentally Strong Tool Agent",
    description="A scalable, decoupled tool-calling agent powered by LangGraph, FastAPI, and free APIs.",
    examples=[
        "What is the current weather in Tokyo?",
        "Convert 150 USD to EUR.",
        "What are the latest news headlines today?",
        "Write me a poem about a robot." # Great for testing our Guardrail prompt!
    ]
)

if __name__ == "__main__":
    demo.launch(server_port=7860)