import gradio as gr
import requests

# The URL of our updated FastAPI backend
API_URL = "http://localhost:8000/api/process-email"

def process_email_ui(raw_email: str):
    """Sends the email to the backend and maps the new Map-Reduce responses."""
    if not raw_email.strip():
        return "Unknown", "❌ Error: Please paste an email.", ""
        
    try:
        response = requests.post(
            API_URL, 
            json={"raw_email": raw_email},
            timeout=60 # Increased timeout to accommodate parallel chunk processing
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("is_valid"):
            return "Unknown", f"❌ Validation Error: {data.get('error')}", ""
            
        # Extract the state variables returned by the backend
        summary = data.get("master_summary", "No summary generated.")
        urgency = data.get("urgency_level", "Unknown")
        draft = data.get("reply_suggestion", "No draft generated.")
        
        # Add visual flair based on the strict urgency categories
        if urgency in ["High", "Critical"]:
            urgency_display = f"🚨 {urgency.upper()}"
        else:
            urgency_display = f"🟢 {urgency}"
             
        return urgency_display, summary, draft
        
    except requests.exceptions.ConnectionError:
        return "Unknown", "❌ Error: Could not connect to the backend. Is FastAPI running?", ""
    except requests.exceptions.Timeout:
        return "Unknown", "❌ Error: Request timed out. The email might be too large or the API is slow.", ""
    except Exception as e:
        return "Unknown", f"❌ Error: {str(e)}", ""

# --- Gradio Interface Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 Enterprise AI Email Assistant")
    gr.Markdown("Powered by a LangGraph Map-Reduce architecture. Paste a massive email thread below to extract a master summary, detect urgency, and draft a context-aware reply without losing details.")
    
    with gr.Row():
        with gr.Column(scale=1):
            email_input = gr.Textbox(
                lines=15, 
                label="Incoming Raw Email (Handles long threads)", 
                placeholder="Paste a long email thread or document here..."
            )
            submit_btn = gr.Button("Process Email", variant="primary")
            
        with gr.Column(scale=1):
            urgency_output = gr.Textbox(lines=1, label="1. Detected Urgency", interactive=False)
            summary_output = gr.Textbox(lines=6, label="2. Master Summary", interactive=False)
            draft_output = gr.Textbox(lines=10, label="3. Reply Suggestion", interactive=False)

    # Wire the button to the function
    submit_btn.click(
        fn=process_email_ui,
        inputs=email_input,
        outputs=[urgency_output, summary_output, draft_output]
    )

if __name__ == "__main__":
    # Launch on port 7860
    demo.launch(server_port=7860)