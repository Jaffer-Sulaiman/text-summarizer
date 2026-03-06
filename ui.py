import gradio as gr
import requests
import time
import concurrent.futures
import re
import tempfile
from fpdf import FPDF

API_URL = "http://localhost:8000/analyze"

def format_list_to_markdown(items, empty_message):
    if not items:
        return f"*{empty_message}*"
    
    markdown_str = ""
    for item in items:
        markdown_str += f"- {item}\n"
    return markdown_str

def create_pdf(title, summary, actions, decisions):
    """Generates a clean PDF from the extracted text and returns the file path."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Helper to strip markdown bolding and emojis to prevent PDF font errors
    def clean_text(text):
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)     # Remove italics
        # Encode to latin-1 to drop unsupported emojis/characters safely
        return text.encode('latin-1', 'replace').decode('latin-1')

    # Clean the inputs
    clean_title = clean_text(title.replace("## 🏷️ ", ""))
    clean_summary = clean_text(summary)
    clean_actions = clean_text(actions)
    clean_decisions = clean_text(decisions)

    # Write Title
    pdf.set_font("helvetica", "B", 16)
    pdf.multi_cell(0, 10, clean_title)
    pdf.ln(5)

    # Write Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.multi_cell(0, 7, clean_summary)
    pdf.ln(5)

    # Write Actions
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Action Items", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.multi_cell(0, 7, clean_actions)
    pdf.ln(5)

    # Write Decisions
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Key Decisions", ln=True)
    pdf.set_font("helvetica", "", 12)
    pdf.multi_cell(0, 7, clean_decisions)

    # Save to a temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

def call_backend(text):
    response = requests.post(API_URL, json={"text": text}, timeout=120)
    if response.status_code != 200:
        error_msg = response.json().get("detail", response.text)
        raise Exception(error_msg)
    return response.json()

def process_text(text):
    if not text.strip():
        # Yield 6 items, keeping the download button hidden
        yield "Untitled", "Please enter some text.", "", "", "⏱️ 0s", gr.update(visible=False)
        return
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(call_backend, text)
        
        while not future.done():
            elapsed = int(time.time() - start_time)
            timer_text = f"**⏱️ Processing: {elapsed}s**"
            yield "⏳ Generating...", "⏳ Reading document...", "⏳ Waiting...", "⏳ Waiting...", timer_text, gr.update(visible=False)
            time.sleep(1) 
        
        try:
            data = future.result()
            
            title = data.get("title", "Untitled Document")
            summary = data.get("summary", "")
            actions_list = data.get("action_items", [])
            decisions_list = data.get("key_decisions", [])
            
            formatted_actions = format_list_to_markdown(actions_list, "No action items found.")
            formatted_decisions = format_list_to_markdown(decisions_list, "No key decisions found.")
            
            final_time = int(time.time() - start_time)
            success_msg = f"**✅ Completed in {final_time}s**"
            
            # Generate the PDF file in the background
            pdf_path = create_pdf(title, summary, formatted_actions, formatted_decisions)
            
            # Yield all results, and make the download button visible with the file path attached!
            yield f"## 🏷️ {title}", summary, formatted_actions, formatted_decisions, success_msg, gr.update(value=pdf_path, visible=True)
            
        except Exception as e:
            yield "❌ Error", f"❌ Backend Error: {str(e)}", "", "", "**❌ Failed**", gr.update(visible=False)

# Build the Gradio Interface
with gr.Blocks(title="AI Text Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 AI Text Analyzer")
    gr.Markdown("Paste a long document, meeting transcript, or email thread below to instantly extract a summary, action items, and key decisions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=20, 
                label="Input Text", 
                placeholder="Paste your long text here..."
            )
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze Text", variant="primary", scale=3)
                timer_display = gr.Markdown("**⏱️ Ready**", label="Timer")
            
        with gr.Column(scale=1):
            # NEW: Add the Download Button at the top of the output column (hidden by default)
            download_btn = gr.DownloadButton("📥 Download Report as PDF", visible=False, variant="secondary")
            
            output_title = gr.Markdown("## 🏷️ Untitled Document")
            
            gr.HTML("<hr>")
            gr.Markdown("### 📝 Executive Summary")
            output_summary = gr.Markdown() 
            
            gr.HTML("<hr>")
            gr.Markdown("### ✅ Action Items")
            output_actions = gr.Markdown() 
            
            gr.HTML("<hr>")
            gr.Markdown("### ⚖️ Key Decisions")
            output_decisions = gr.Markdown() 
            
    # Wire the button to expect 6 outputs now
    analyze_btn.click(
        fn=process_text,
        inputs=input_text,
        outputs=[output_title, output_summary, output_actions, output_decisions, timer_display, download_btn]
    )

if __name__ == "__main__":
    demo.queue().launch(server_port=7860)