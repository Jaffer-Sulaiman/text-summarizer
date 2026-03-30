import gradio as gr
import requests
import uuid

API_URL = "http://localhost:8000/api/chat"
# Generate a unique session ID for this browser tab
SESSION_ID = str(uuid.uuid4())

def chat_with_agent(message, history):
    """Sends the message and thread ID to FastAPI."""
    try:
        payload = {
            "message": message,
            "thread_id": SESSION_ID
        }
        response = requests.post(API_URL, json=payload, timeout=45)
        response.raise_for_status()
        
        return response.json()["response"]
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to the FastAPI backend."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Build the Chat UI
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="📧 Email Drafting Copilot",
    description="Paste an email to start. The agent will process it and draft a reply. You can then chat with the agent to refine and adjust the draft.",
    examples=["Hi support, my order #999 was supposed to arrive yesterday but I haven't seen it yet. Can you help?", "This is a cold sales pitch offering you SEO services!"],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(server_port=7860)