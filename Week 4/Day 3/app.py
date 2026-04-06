import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
MODEL_NAME = "all-MiniLM-L6-v2"
app = FastAPI(title="Day 3: Retrieval Evaluation API")

print("Loading Embedding Model...")
model = SentenceTransformer(MODEL_NAME)
dim = model.get_sentence_embedding_dimension()

# Initialize FAISS and Document Store
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
doc_store = {}
current_id = 0

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# --- 2. Pydantic Models for API ---
class SeedDocument(BaseModel):
    text: str
    source: str

class SeedRequest(BaseModel):
    documents: list[SeedDocument]

# --- 3. API ENDPOINTS ---
@app.post("/api/seed")
async def seed_database(request: SeedRequest):
    """Instantly injects raw text documents into the vector database for testing."""
    global current_id
    
    # Clear existing data for a fresh test run
    faiss_index.reset()
    doc_store.clear()
    current_id = 0
    
    total_chunks = 0
    
    for doc in request.documents:
        chunks = text_splitter.split_text(doc.text)
        num_chunks = len(chunks)
        
        if num_chunks == 0:
            continue
            
        embeddings = model.encode(chunks)
        embeddings_np = np.array(embeddings).astype('float32')
        
        start_id = current_id
        end_id = start_id + num_chunks
        ids_np = np.arange(start_id, end_id).astype('int64')
        
        faiss_index.add_with_ids(embeddings_np, ids_np)
        
        for i, chunk_id in enumerate(range(start_id, end_id)):
            doc_store[chunk_id] = {
                "text": chunks[i],
                "source": doc.source
            }
            
        current_id = end_id
        total_chunks += num_chunks
        
    return {"status": "success", "message": f"Seeded {len(request.documents)} documents into {total_chunks} chunks."}

@app.get("/api/search")
async def search_database(query: str, top_k: int = 3):
    """Searches the FAISS index and returns the top K results."""
    if faiss_index.ntotal == 0:
         return {"results": []}
         
    query_np = np.array(model.encode([query])).astype('float32')
    distances, indices = faiss_index.search(query_np, top_k)
    
    results = []
    for i, chunk_id in enumerate(indices[0]):
        if chunk_id != -1:
            data = doc_store.get(int(chunk_id))
            results.append({
                "source": data["source"],
                "text": data["text"],
                "distance": float(distances[0][i])
            })
            
    return {"results": results}

if __name__ == "__main__":
    print("Starting API Server... Available at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)