import gradio as gr
import requests
import time
import concurrent.futures
import re
import tempfile
from fpdf import FPDF
import pypdf

API_URL = "http://localhost:8000/analyze"

def format_list_to_markdown(items, empty_message):
    if not items:
        return f"*{empty_message}*"
    markdown_str = ""
    for item in items:
        markdown_str += f"- {item}\n"
    return markdown_str

def create_pdf(title, summary, tasks, risks, decisions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    def clean_text(text):
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) 
        text = re.sub(r'\*(.*?)\*', r'\1', text)     
        return text.encode('latin-1', 'replace').decode('latin-1')

    clean_title = clean_text(title.replace("## 🏷️ ", ""))
    
    pdf.set_font("helvetica", "B", 16)
    pdf.multi_cell(0, 10, clean_title)
    pdf.ln(5)

    sections = [
        ("Executive Summary", summary),
        ("Tasks", tasks),
        ("Risks & Blockers", risks),
        ("Decision Points", decisions)
    ]

    for heading, content in sections:
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, heading, ln=True)
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 7, clean_text(content))
        pdf.ln(5)

    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name

def process_uploaded_file(file_obj):
    if file_obj is None: return ""
    file_path = file_obj.name
    extracted_text = ""
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        elif file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text: extracted_text += text + "\n"
        return extracted_text
    except Exception as e:
        return f"Error reading file: {str(e)}"

def call_backend(text):
    response = requests.post(API_URL, json={"text": text}, timeout=120)
    if response.status_code != 200:
        raise Exception(response.json().get("detail", response.text))
    return response.json()

def process_text(text):
    if not text.strip():
        # Yielding 7 items now
        yield "Untitled", "Please enter some text.", "", "", "", "⏱️ 0s", gr.update(visible=False)
        return
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(call_backend, text)
        
        while not future.done():
            elapsed = int(time.time() - start_time)
            timer_text = f"**⏱️ Processing: {elapsed}s**"
            yield "⏳ Generating...", "⏳ Reading document...", "⏳ Waiting...", "⏳ Waiting...", "⏳ Waiting...", timer_text, gr.update(visible=False)
            time.sleep(1) 
        
        try:
            data = future.result()
            title = data.get("title", "Title not found")
            summary = data.get("summary", "Summary not in context")
            
            # Apply strict fallbacks
            formatted_tasks = format_list_to_markdown(data.get("tasks", []), "Tasks not found")
            formatted_risks = format_list_to_markdown(data.get("risks", []), "Risks not found")
            formatted_decisions = format_list_to_markdown(data.get("decision_points", []), "Decision points not found") 
            
            final_time = int(time.time() - start_time)
            success_msg = f"**✅ Completed in {final_time}s**"
            
            pdf_path = create_pdf(title, summary, formatted_tasks, formatted_risks, formatted_decisions)
            
            yield f"## 🏷️ {title}", summary, formatted_tasks, formatted_risks, formatted_decisions, success_msg, gr.update(value=pdf_path, visible=True)
            
        except Exception as e:
            yield "❌ Error", f"❌ Backend Error: {str(e)}", "", "", "", "**❌ Failed**", gr.update(visible=False)

with gr.Blocks(title="Meeting Intelligence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Meeting Intelligence Tool")
    gr.Markdown("Upload a transcript to instantly extract the summary, assigned tasks, project risks, and key decisions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            document_upload = gr.File(label="Upload Transcript (.pdf or .txt)", file_types=[".pdf", ".txt"], type="filepath")
            input_text = gr.Textbox(lines=12, label="Input Text", placeholder="Upload a file above, or paste your text here...")
            with gr.Row():
                analyze_btn = gr.Button("Analyze Transcript", variant="primary", scale=3)
                timer_display = gr.Markdown("**⏱️ Ready**", label="Timer")
            
        with gr.Column(scale=1):
            download_btn = gr.DownloadButton("📥 Download Report as PDF", visible=False, variant="secondary")
            output_title = gr.Markdown("## 🏷️ Untitled Document")
            
            gr.HTML("<hr>")
            gr.Markdown("### 📝 Executive Summary")
            output_summary = gr.Markdown() 
            
            gr.HTML("<hr>")
            gr.Markdown("### ✅ Tasks")
            output_tasks = gr.Markdown() 
            
            gr.HTML("<hr>")
            gr.Markdown("### ⚠️ Risks & Blockers")
            output_risks = gr.Markdown() 
            
            gr.HTML("<hr>")
            gr.Markdown("### ⚖️ Decision Points")
            output_decisions = gr.Markdown() 
            
    document_upload.change(fn=process_uploaded_file, inputs=document_upload, outputs=input_text)
            
    analyze_btn.click(
        fn=process_text,
        inputs=input_text,
        outputs=[output_title, output_summary, output_tasks, output_risks, output_decisions, timer_display, download_btn]
    ) 

if __name__ == "__main__":
    demo.queue().launch(server_port=7860)