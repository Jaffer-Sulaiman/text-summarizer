import os
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ==========================================
# Phase 1: Setup & Initialization
# ==========================================

# Ensure you have set your GOOGLE_API_KEY in your environment variables
# os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

# Initialize Gemini LLM (using flash for speed/cost efficiency in RAG)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Or gemini-1.5-flash depending on your region/access
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize Google Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Persistent ChromaDB
# We use a single collection and rely on metadata filtering for specific transcripts
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "meeting_transcripts"

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)

# ==========================================
# Phase 2: Ingestion Logic
# ==========================================

def process_and_store_document(
    input_data: str, 
    input_type: str, 
    doc_name: str
) -> str:
    """
    Processes raw text or files, chunks them, and stores them in ChromaDB.
    Returns the unique doc_id for UI state tracking.
    """
    docs: List[Document] = []
    
    # 1. Load Data based on input type
    if input_type == "text":
        # Direct paste
        docs = [Document(page_content=input_data)]
    elif input_type == "txt_file":
        # input_data is the file path from Gradio
        loader = TextLoader(input_data)
        docs = loader.load()
    elif input_type == "pdf_file":
        # input_data is the file path from Gradio
        loader = PyPDFLoader(input_data)
        docs = loader.load()
    else:
        raise ValueError("Unsupported input type. Use 'text', 'txt_file', or 'pdf_file'.")

    # 2. Chunk the documents
    # Using generous overlap to prevent cutting off conversational context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # 3. Tag with Metadata and Store
    # We generate a unique ID for this specific transcript upload
    doc_id = str(uuid.uuid4())
    
    for split in splits:
        split.metadata["doc_id"] = doc_id
        split.metadata["doc_name"] = doc_name

    # Add to ChromaDB
    vectorstore.add_documents(documents=splits)
    
    return doc_id

def get_available_transcripts() -> dict:
    """
    Retrieves unique transcripts available in the DB for the Gradio dropdown.
    Returns a dict mapping doc_name -> doc_id.
    """
    try:
        # Fetch metadata from ChromaDB
        data = vectorstore.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        # Deduplicate to get unique transcripts
        transcripts = {}
        for meta in metadatas:
            if meta and "doc_name" in meta and "doc_id" in meta:
                transcripts[meta["doc_name"]] = meta["doc_id"]
                
        return transcripts
    except Exception as e:
        print(f"Error fetching transcripts: {e}")
        return {}