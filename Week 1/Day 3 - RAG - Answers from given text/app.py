import os
import gradio as gr

# Import our backend logic
# Assuming Phase 1 & 2 are in rag_core.py and Phase 3 is in graph_core.py
from rag_core import process_and_store_document, get_available_transcripts
from graph_core import app_graph

# ==========================================
# UI Callback Functions
# ==========================================

def load_transcripts_for_dropdown():
    """Fetches transcripts from ChromaDB to populate the dropdown."""
    transcripts = get_available_transcripts()
    # Gradio dropdown expects a list of tuples: (Display Name, Value)
    choices = [(name, doc_id) for name, doc_id in transcripts.items()]
    return gr.update(choices=choices)

def handle_upload(text_input, file_input, doc_name):
    """Processes the input and stores it in the vector DB."""
    if not doc_name.strip():
        return "⚠️ Error: Please provide a distinct name for this transcript.", gr.update()
    
    try:
        if text_input.strip():
            doc_id = process_and_store_document(text_input, "text", doc_name)
        elif file_input is not None:
            # Gradio file_input returns a file path string
            ext = os.path.splitext(file_input)[1].lower()
            if ext == ".txt":
                doc_id = process_and_store_document(file_input, "txt_file", doc_name)
            elif ext == ".pdf":
                doc_id = process_and_store_document(file_input, "pdf_file", doc_name)
            else:
                return "⚠️ Error: Unsupported file format. Use .txt or .pdf.", gr.update()
        else:
            return "⚠️ Error: Please provide either direct text or upload a file.", gr.update()

        # Return a success message and update the dropdown with the new file
        return f"✅ Successfully ingested: {doc_name}", load_transcripts_for_dropdown()
        
    except Exception as e:
        return f"❌ Error during ingestion: {str(e)}", gr.update()

def chat_interface(user_message, history, transcript_id):
    """Invokes the LangGraph state machine and formats the response."""
    if not transcript_id:
        return "⚠️ Please select a transcript from the sidebar before chatting."
    
    # Initialize the graph state
    initial_state = {
        "question": user_message, 
        "transcript_id": transcript_id
    }
    
    # Invoke our compiled LangGraph
    result = app_graph.invoke(initial_state)
    
    # Extract outputs
    answer = result.get("answer", "I encountered an error generating a response.")
    sources = result.get("sources", [])
    
    # Format the final string with expandable citations (using markdown blockquotes)
    final_output = answer
    if sources:
        final_output += "\n\n---\n**Sources used:**\n"
        for i, src in enumerate(sources, 1):
            # Clean up newlines for cleaner rendering
            clean_src = src.replace('\n', ' ').strip()
            final_output += f"> *{i}. {clean_src}*\n\n"
            
    return final_output

# ==========================================
# Gradio Layout Definition
# ==========================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Scalable RAG Transcript Agent")
    gr.Markdown("Upload meeting transcripts and chat with them. Powered by LangGraph, ChromaDB, and Gemini.")

    with gr.Row():
        # --- SIDEBAR (Ingestion & Configuration) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload New Transcript")
            doc_name_input = gr.Textbox(label="Transcript Name (Required)", placeholder="e.g., Q3 Marketing Sync")
            
            with gr.Tabs():
                with gr.TabItem("Upload File"):
                    file_upload = gr.File(label="Upload .txt or .pdf", file_types=[".txt", ".pdf"])
                with gr.TabItem("Paste Text"):
                    text_paste = gr.Textbox(label="Paste raw transcript text here", lines=10)
            
            upload_btn = gr.Button("Process & Store", variant="primary")
            upload_status = gr.Markdown()
            
            gr.Markdown("---")
            
            gr.Markdown("### 2. Select Active Transcript")
            transcript_dropdown = gr.Dropdown(
                label="Choose a transcript to chat with", 
                choices=[], 
                interactive=True
            )
            refresh_btn = gr.Button("🔄 Refresh List", size="sm")
            
        # --- MAIN PANEL (Chat) ---
        with gr.Column(scale=3):
            gr.Markdown("### 3. Chat with your Data")
            # We use Chatbot for the history display, and a textbox for the input
            chatbot = gr.Chatbot(height=500)
            msg_input = gr.Textbox(label="Ask a question about the selected transcript...", placeholder="What were the key action items?")
            clear_btn = gr.ClearButton([msg_input, chatbot])

    # ==========================================
    # Event Wiring
    # ==========================================
    
    # On app load, populate the dropdown
    demo.load(fn=load_transcripts_for_dropdown, outputs=transcript_dropdown)
    
    # Refresh button wiring
    refresh_btn.click(fn=load_transcripts_for_dropdown, outputs=transcript_dropdown)
    
    # Upload button wiring
    upload_btn.click(
        fn=handle_upload,
        inputs=[text_paste, file_upload, doc_name_input],
        outputs=[upload_status, transcript_dropdown]
    )
    
    # Chat wiring
    def user_submit(user_msg, chat_history, active_id):
        """Helper to append to history and get bot response."""
        # Append user message immediately
        chat_history = chat_history + [[user_msg, None]]
        # Get bot response
        bot_reply = chat_interface(user_msg, chat_history, active_id)
        # Update history with bot response
        chat_history[-1][1] = bot_reply
        return "", chat_history

    msg_input.submit(
        fn=user_submit, 
        inputs=[msg_input, chatbot, transcript_dropdown], 
        outputs=[msg_input, chatbot]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(debug=True)