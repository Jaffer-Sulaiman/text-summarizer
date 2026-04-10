import gradio as gr
import requests
import json

API_URL = "http://localhost:8000"

def upload_document(file_obj):
    """Uploads a user document to the FastAPI backend."""
    if file_obj is None:
        return "⚠️ No file selected."
        
    try:
        # Prepare the file payload for multipart POST
        files = {"file": (file_obj.name, open(file_obj.name, "rb"))}
        response = requests.post(f"{API_URL}/upload", files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "duplicate":
                 return f"⚠️ {data.get('message')}"
            return f"✅ **Success**: {data.get('message')} ({data.get('chunks_added', 0)} chunks inserted)"
        else:
            error_data = response.json()
            return f"❌ **Error**: {error_data.get('detail', response.text)}"
            
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error**: Make sure the backend API (api.py) is running on port 8000."
    except Exception as e:
        return f"❌ **Error**: {str(e)}"

def extract_text(content):
    """Extract string uniformly from Gradio 6's complex content types."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return " ".join(texts)
    elif isinstance(content, tuple) or isinstance(content, list):
        return str(content[0]) if len(content) > 0 else ""
    return str(content)

def format_history(gradio_history):
    """Convert Gradio's messages list to API format."""
    api_history = []
    for msg in gradio_history:
        # Map 'assistant' from Gradio to 'ai' for our backend
        role = "ai" if msg["role"] == "assistant" else "user"
        content_str = extract_text(msg["content"])
        api_history.append({"role": role, "content": content_str})
    return api_history

def chat_interaction(user_message, chat_history):
    """Handle chat interaction between User and RAG API."""
    if not user_message.strip():
        return "", chat_history
        
    # Append the user's message
    chat_history.append({"role": "user", "content": user_message})
    # Append a temporary AI loading message
    chat_history.append({"role": "assistant", "content": "⏳ Generating response..."})
    yield "", chat_history
    
    # History payload should exclude the 2 newly added messages
    prior_history = format_history(chat_history[:-2])
    
    payload = {
        "query": user_message,
        "history": prior_history
    }
    
    try:
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer received.")
            chat_history[-1] = {"role": "assistant", "content": answer}
        else:
            try:
                error_detail = response.json().get("detail", response.text)
                chat_history[-1] = {"role": "assistant", "content": f"❌ **Backend Error**: {error_detail}"}
            except:
                chat_history[-1] = {"role": "assistant", "content": f"❌ **Error**: {response.text}"}
                
    except requests.exceptions.ConnectionError:
        chat_history[-1] = {"role": "assistant", "content": "❌ **Connection Error**: Cannot reach API at :8000."}
    except Exception as e:
         chat_history[-1] = {"role": "assistant", "content": f"❌ **Error**: {str(e)}"}
         
    yield "", chat_history

# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================
with gr.Blocks(title="Robust RAG Assistant") as demo:
    gr.Markdown("# 🧠 Robust RAG Assistant")
    gr.Markdown("Upload documents into the local context database, and engage in a grounded conversation with conversational memory.")
    
    with gr.Tabs():
        with gr.Tab("💬 Chat Interface"):
            chatbot = gr.Chatbot(height=500, label="Agent RAG Chat")
            
            with gr.Row():
                msg_input = gr.Textbox(placeholder="Ask a question...", scale=4, show_label=False)
                send_btn = gr.Button("Send", variant="primary", scale=1)
                
            clear_btn = gr.ClearButton([msg_input, chatbot], value="🗑️ Clear Chat History")
            
            # Event hook
            msg_input.submit(chat_interaction, [msg_input, chatbot], [msg_input, chatbot])
            send_btn.click(chat_interaction, [msg_input, chatbot], [msg_input, chatbot])
            
        with gr.Tab("📄 Knowledge Base Manager"):
            gr.Markdown("### Document Ingestion")
            gr.Markdown("Supports `.pdf` and `.txt`. Documents are hashed to prevent duplicate indexing.")
            
            upload_box = gr.File(label="Upload Document", file_types=[".pdf", ".txt"])
            upload_status = gr.Markdown("⏳ **Status**: Waiting for upload...")
            
            upload_box.upload(upload_document, inputs=[upload_box], outputs=[upload_status])
            
if __name__ == "__main__":
    demo.queue().launch(server_port=7860, theme=gr.themes.Soft())
