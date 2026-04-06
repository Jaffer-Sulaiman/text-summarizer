import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. SETUP & CONFIGURATION 
# ==========================================

MODEL_NAMES = {
    "MiniLM (Fast)": "all-MiniLM-L6-v2",
    "BGE Small (High Quality)": "BAAI/bge-small-en-v1.5"
}

app = FastAPI(title="Day 2: FAISS Multi-Model Semantic Search")

print("Loading Models and initializing FAISS indexes...")
models = {}
faiss_indexes = {}
doc_stores = {} # Our in-memory database to hold the actual text
next_id = {}    # Keeps track of the integer IDs for FAISS

for name, hf_id in MODEL_NAMES.items():
    model = SentenceTransformer(hf_id)
    models[name] = model
    
    dim = model.get_sentence_embedding_dimension()
    base_index = faiss.IndexFlatL2(dim)
    faiss_indexes[name] = faiss.IndexIDMap(base_index)
    
    doc_stores[name] = {}
    next_id[name] = 0

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# ==========================================
# 2. FASTAPI BACKEND LOGIC
# ==========================================

def process_and_embed_document(file_obj, filename: str):
    """Extracts text, chunks it, and indexes it in FAISS and our dict."""
    try:
        reader = PdfReader(file_obj)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
            
        if not raw_text.strip():
            return False, "Could not extract text. The PDF might be scanned images."

        chunks = text_splitter.split_text(raw_text)
        if not chunks:
            return False, "Document is empty after chunking."

        num_chunks = len(chunks)

        for model_name, model in models.items():
            embeddings = model.encode(chunks)
            embeddings_np = np.array(embeddings).astype('float32')
            
            start_id = next_id[model_name]
            end_id = start_id + num_chunks
            ids_np = np.arange(start_id, end_id).astype('int64')
            
            faiss_indexes[model_name].add_with_ids(embeddings_np, ids_np)
            
            for i, chunk_id in enumerate(range(start_id, end_id)):
                doc_stores[model_name][chunk_id] = {
                    "text": chunks[i],
                    "source": filename,
                    "chunk_index": i
                }
            
            next_id[model_name] = end_id
            
        return True, f"Successfully processed '{filename}' into {num_chunks} chunks."
    except Exception as e:
        return False, str(e)


@app.post("/api/upload")
async def api_upload_file(file: UploadFile = File(...)):
    success, message = process_and_embed_document(file.file, file.filename)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "success", "message": message}


@app.post("/api/search")
async def api_search(query: str, model_name: str, top_k: int = 3):
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model selected.")
        
    model = models[model_name]
    index = faiss_indexes[model_name]
    doc_store = doc_stores[model_name]
    
    if index.ntotal == 0:
         return {"results": [], "message": "The index is empty."}
    
    query_np = np.array(model.encode([query])).astype('float32')
    distances, indices = index.search(query_np, top_k)
    
    formatted_results = []
    for i, chunk_id in enumerate(indices[0]):
        if chunk_id == -1: 
            continue
            
        doc_data = doc_store.get(int(chunk_id))
        if doc_data:
            formatted_results.append({
                "source": doc_data["source"],
                "text": doc_data["text"],
                "distance": round(float(distances[0][i]), 4) 
            })
            
    return {"results": formatted_results}

# ==========================================
# 3. GRADIO FRONTEND UI
# ==========================================

def gradio_upload(filepath):
    if filepath is None:
        return "Please upload a file."
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        success, message = process_and_embed_document(f, filename)
    return message

def gradio_search(query, model_name):
    if not query:
        return "Please enter a search query."
    
    model = models[model_name]
    index = faiss_indexes[model_name]
    doc_store = doc_stores[model_name]
    
    if index.ntotal == 0:
        return "The database is empty. Please ingest a document first."
        
    query_np = np.array(model.encode([query])).astype('float32')
    distances, indices = index.search(query_np, 3)
    
    output = ""
    for i, chunk_id in enumerate(indices[0]):
        if chunk_id != -1:
            doc_data = doc_store.get(int(chunk_id))
            dist = distances[0][i]
            output += f"### Result {i+1} (Source: {doc_data['source']} | L2 Distance: {dist:.4f})\n"
            output += f"> {doc_data['text']}\n\n---\n"
        
    return output if output else "No valid results found."

with gr.Blocks(title="FAISS Semantic Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Day 2: FAISS Multi-Model Semantic Search")
    
    with gr.Tabs():
        with gr.Tab("1. Ingest Documents"):
            gr.Markdown("Upload a PDF. Text is mapped to FAISS integer IDs and stored in memory.")
            file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            upload_btn = gr.Button("Process & Embed Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            upload_btn.click(fn=gradio_upload, inputs=file_input, outputs=upload_status)
            
        with gr.Tab("2. Semantic Search"):
            gr.Markdown("Search the FAISS index. Lower L2 distances mean closer matches.")
            with gr.Row():
                query_input = gr.Textbox(label="Search Query", placeholder="Enter your question here...", scale=3)
                model_dropdown = gr.Dropdown(choices=list(MODEL_NAMES.keys()), value=list(MODEL_NAMES.keys())[0], label="Embedding Model", scale=1)
            
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Markdown(label="Results")
            search_btn.click(fn=gradio_search, inputs=[query_input, model_dropdown], outputs=search_results)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    print("Starting Day 2 Server... Access the UI at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)