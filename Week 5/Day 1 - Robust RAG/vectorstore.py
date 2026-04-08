import os
import hashlib
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Use a local directory for persistency
DB_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")

class VectorStoreManager:
    def __init__(self):
        # Initialize HuggingFace embeddings (Sentence Transformers)
        # BAAI/bge-small-en-v1.5 is a very fast and capable open source model.
        # all-MiniLM-L6-v2 is also good, but bge-small is generally better for RAG.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize the Chroma store
        self.vectorstore = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=DB_DIR
        )
        
        # Initialize Text Splitter for chunking large docs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def _get_file_hash(self, text: str) -> str:
        """Generate MD5 hash for the text to prevent duplicates."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _is_duplicate(self, file_hash: str) -> bool:
        """Check if a file hash already exists in the vector store."""
        try:
            # We fetch 1 document with the matching hash from metadata
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 1, "filter": {"file_hash": file_hash}}
            )
            docs = retriever.invoke("test")
            return len(docs) > 0
        except Exception:
            return False

    def is_empty(self) -> bool:
        """Check if the vector store is completely empty."""
        try:
            count = self.vectorstore._collection.count()
            return count == 0
        except Exception:
            return True

    def ingest_document(self, text: str, source_name: str) -> Dict[str, Any]:
        """
        Chunk and ingest a document. 
        Returns status, including chunk counts or duplicate alerts.
        """
        # Validate text
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided."}

        # Generate hash
        file_hash = self._get_file_hash(text)
        
        # Check for duplicates using Chroma collection
        existing_docs = self.vectorstore.get(where={"file_hash": file_hash})
        if existing_docs and len(existing_docs["ids"]) > 0:
             return {
                 "status": "duplicate", 
                 "message": f"Document '{source_name}' already exists in the database.",
                 "chunks_added": 0
             }

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        if not chunks:
             return {"status": "error", "message": "Failed to extract chunks from document."}
             
        # Create full documents with metadata
        documents = [
            Document(page_content=chunk, metadata={"source": source_name, "file_hash": file_hash})
            for chunk in chunks
        ]
        
        # Upsert to Chroma
        self.vectorstore.add_documents(documents)
        
        return {
            "status": "success",
            "message": f"Successfully ingested '{source_name}'.",
            "chunks_added": len(chunks)
        }

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """Retrieve the top_k most similar chunks for a given query."""
        if self.is_empty():
            return []
            
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)

# Singleton instance
vector_manager = VectorStoreManager()
