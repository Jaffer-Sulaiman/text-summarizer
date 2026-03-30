import gradio as gr
import requests

# The URL of our FastAPI backend
API_URL = "http://localhost:8000/api/process-email"

def process_email_ui(raw_email: str):
    """Sends the email to the backend and formats the response for the UI."""
    if not raw_email.strip():
        return "Error: Please enter an email.", "", ""
        
    try:
        response = requests.post(
            API_URL, 
            json={"raw_email": raw_email},
            timeout=30 # Good practice to avoid hanging the UI
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("is_valid"):
            return f"❌ Validation Error: {data.get('error')}", "", ""
            
        if data.get("intent") in ["spam", "other"]:
             return (
                 f"Summary: {data.get('summary')}", 
                 f"🛑 Intent: {data.get('intent').upper()} (Filtered)", 
                 "No draft generated for this intent category."
             )
             
        return (
            data.get("summary", "No summary generated."),
            data.get("intent", "Unknown"),
            data.get("draft", "No draft generated.")
        )
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: Could not connect to the backend. Is FastAPI running?", "", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# --- Gradio Interface Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 Multi-Step Email Processing Agent")
    gr.Markdown("Pastes an email below. The agent will check it, summarize it, identify the intent using Gemini, and draft a context-aware response.")
    
    with gr.Row():
        with gr.Column(scale=1):
            email_input = gr.Textbox(
                lines=10, 
                label="Incoming Raw Email", 
                placeholder="Paste the email content here..."
            )
            submit_btn = gr.Button("Process Email", variant="primary")
            
        with gr.Column(scale=1):
            summary_output = gr.Textbox(lines=3, label="1. Extracted Summary", interactive=False)
            intent_output = gr.Textbox(lines=1, label="2. Identified Intent", interactive=False)
            draft_output = gr.Textbox(lines=8, label="3. Generated Draft Response", interactive=False)

    # Wire the button to the function
    submit_btn.click(
        fn=process_email_ui,
        inputs=email_input,
        outputs=[summary_output, intent_output, draft_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)