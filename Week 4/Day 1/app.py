import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Define the models we want to use
MODEL_NAMES = {
    "MiniLM (Fast)": "all-MiniLM-L6-v2",
    "BGE Small (High Quality)": "BAAI/bge-small-en-v1.5"
}

print("Loading Embedding Models... (This might take a minute on first run)")
models = {name: SentenceTransformer(hf_id) for name, hf_id in MODEL_NAMES.items()}

# Initialize ChromaDB (Persistent local storage)
chroma_client = chromadb.PersistentClient(path="./chroma_data")

# Create a collection for each model
collections = {}
for name in MODEL_NAMES.keys():
    # Chroma collection names must be alphanumeric and underscores
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    collections[name] = chroma_client.get_or_create_collection(name=f"collection_{safe_name}")

app = FastAPI(title="Multi-Model Semantic Search API")

# Setup Text Splitter (500 chars with 50 char overlap handles context limits well)
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
    """Extracts text, chunks it, and embeds it across all models."""
    try:
        # Extract Text
        reader = PdfReader(file_obj)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
            
        if not raw_text.strip():
            return False, "Could not extract text. The PDF might be scanned images."

        # Chunk Text
        chunks = text_splitter.split_text(raw_text)
        if not chunks:
            return False, "Document is empty after chunking."

        # Generate unique IDs for these chunks
        doc_ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # Embed and Store for EVERY model
        for model_name, model in models.items():
            print(f"Embedding {len(chunks)} chunks using {model_name}...")
            # Generate embeddings
            embeddings = model.encode(chunks).tolist()
            
            # Upsert to specific ChromaDB collection
            collections[model_name].upsert(
                ids=doc_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadata
            )
            
        return True, f"Successfully processed '{filename}' into {len(chunks)} chunks across {len(models)} models."
    except Exception as e:
        return False, str(e)


@app.post("/api/upload")
async def api_upload_file(file: UploadFile = File(...)):
    """FastAPI endpoint for file uploads"""
    success, message = process_and_embed_document(file.file, file.filename)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "success", "message": message}


@app.post("/api/search")
async def api_search(query: str, model_name: str, top_k: int = 3):
    """FastAPI endpoint for semantic search"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Invalid model selected.")
        
    model = models[model_name]
    collection = collections[model_name]
    
    # 1. Embed the search query
    query_embedding = model.encode([query]).tolist()
    
    # 2. Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # 3. Format results
    formatted_results = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            formatted_results.append({
                "source": meta['source'],
                "text": doc_text,
                "distance": round(dist, 4)
            })
            
    return {"results": formatted_results}

# ==========================================
# 3. GRADIO FRONTEND UI
# ==========================================

# Gradio Wrapper Functions (interacting directly with python logic for efficiency)
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
    
    # Re-use the logic from the API endpoint
    model = models[model_name]
    collection = collections[model_name]
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    if not results['documents'] or not results['documents'][0]:
        return "No results found."
        
    output = ""
    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        source = results['metadatas'][0][i]['source']
        score = results['distances'][0][i]
        output += f"### Result {i+1} (Source: {source} | Distance: {score:.4f})\n"
        output += f"> {text}\n\n---\n"
        
    return output

# Build the UI
with gr.Blocks(title="Semantic Search Architecture", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Multi-Model Document Semantic Search")
    
    with gr.Tabs():
        with gr.Tab("1. Ingest Documents"):
            gr.Markdown("Upload a PDF. The system will chunk the text and create embeddings for **all** loaded models.")
            file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            upload_btn = gr.Button("Process & Embed Document", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False)
            
            upload_btn.click(fn=gradio_upload, inputs=file_input, outputs=upload_status)
            
        with gr.Tab("2. Semantic Search"):
            gr.Markdown("Search your database. Select which model's embedding space you want to query.")
            with gr.Row():
                query_input = gr.Textbox(label="Search Query", placeholder="What are the main concepts in the document?", scale=3)
                model_dropdown = gr.Dropdown(choices=list(MODEL_NAMES.keys()), value=list(MODEL_NAMES.keys())[0], label="Embedding Model", scale=1)
            
            search_btn = gr.Button("Search", variant="primary")
            search_results = gr.Markdown(label="Results")
            
            search_btn.click(fn=gradio_search, inputs=[query_input, model_dropdown], outputs=search_results)

# Mount Gradio app onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    print("Starting server... Access the UI at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)