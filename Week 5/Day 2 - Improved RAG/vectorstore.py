import os
import re
import hashlib
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Use a local directory for persistency — scoped to Day 2
DB_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")

# ==============================================================================
# CONSTANTS
# ==============================================================================
MIN_CHUNK_CHARS = 50        # Discard chunks shorter than this (headers, whitespace, etc.)
DENSE_CHUNK_SIZE = 512      # Used for short-sentence, data-dense documents
PROSE_CHUNK_SIZE = 1500     # Used for long-sentence, flowing prose documents
CHUNK_OVERLAP = 150         # Consistent overlap regardless of chunk size
# Sentence boundary is detected when avg words-per-sentence exceeds this threshold
PROSE_SENTENCE_THRESHOLD = 18


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = Chroma(
            collection_name="rag_collection_v2",
            embedding_function=self.embeddings,
            persist_directory=DB_DIR
        )

        # Separators split on natural sentence/thought boundaries, never mid-word
        self._separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]

    # --------------------------------------------------------------------------
    # ADAPTIVE CHUNKING
    # --------------------------------------------------------------------------
    def _estimate_doc_type(self, text: str) -> str:
        """
        Heuristic: estimate document style by average words-per-sentence.
        Short sentences → dense/tabular/transcript → use smaller chunks.
        Long sentences  → flowing prose/reports    → use larger chunks.
        """
        # Sample first 3000 chars to stay fast
        sample = text[:3000]
        sentences = re.split(r"[.!?]\s+", sample)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return "prose"
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        return "dense" if avg_words < PROSE_SENTENCE_THRESHOLD else "prose"

    def _make_splitter(self, doc_type: str) -> RecursiveCharacterTextSplitter:
        chunk_size = DENSE_CHUNK_SIZE if doc_type == "dense" else PROSE_CHUNK_SIZE
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=CHUNK_OVERLAP,
            separators=self._separators,
        )

    def _filter_chunks(self, chunks: List[str]) -> List[str]:
        """
        Post-split quality filter:
        1. Discard chunks shorter than MIN_CHUNK_CHARS (headings, page numbers, etc.)
        2. Discard chunks where alphabetic chars are < 40% of content
           (catches whitespace pages, separator lines, footer debris).
        """
        clean = []
        for chunk in chunks:
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            alpha_ratio = sum(c.isalpha() for c in chunk) / max(len(chunk), 1)
            if alpha_ratio < 0.40:
                continue
            clean.append(chunk)
        return clean

    # --------------------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------------------
    def _get_file_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def is_empty(self) -> bool:
        try:
            return self.vectorstore._collection.count() == 0
        except Exception:
            return True

    def get_collection_count(self) -> int:
        """Return total number of chunks currently in the store."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    # --------------------------------------------------------------------------
    # INGEST
    # --------------------------------------------------------------------------
    def ingest_document(self, text: str, source_name: str) -> Dict[str, Any]:
        """
        Chunk and ingest a document with adaptive sizing.
        Returns status, chunk count, doc_type detected, and avg_chunk_tokens.
        """
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided."}

        file_hash = self._get_file_hash(text)

        # Duplicate check
        existing = self.vectorstore.get(where={"file_hash": file_hash})
        if existing and len(existing["ids"]) > 0:
            return {
                "status": "duplicate",
                "message": f"'{source_name}' already exists in the knowledge base.",
                "chunks_added": 0,
            }

        # Adaptive chunking
        doc_type = self._estimate_doc_type(text)
        splitter = self._make_splitter(doc_type)
        raw_chunks = splitter.split_text(text)

        # Quality filter
        chunks = self._filter_chunks(raw_chunks)

        if not chunks:
            return {
                "status": "error",
                "message": "Could not extract usable chunks. The document may be blank or contain only non-text content.",
            }

        # Estimate average tokens (rough: 1 token ≈ 4 chars)
        avg_chunk_tokens = int(sum(len(c) for c in chunks) / len(chunks) / 4)

        documents = [
            Document(
                page_content=chunk,
                metadata={"source": source_name, "file_hash": file_hash},
            )
            for chunk in chunks
        ]

        self.vectorstore.add_documents(documents)

        return {
            "status": "success",
            "message": f"Successfully ingested '{source_name}'.",
            "chunks_added": len(chunks),
            "chunks_discarded": len(raw_chunks) - len(chunks),
            "doc_type_detected": doc_type,
            "avg_chunk_tokens": avg_chunk_tokens,
        }

    # --------------------------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """Retrieve top_k most similar chunks. top_k already validated by caller."""
        if self.is_empty():
            return []
        # Cap top_k to collection size to avoid ChromaDB errors on small DBs
        capped_k = min(top_k, self.get_collection_count())
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": capped_k})
        return retriever.invoke(query)


# Singleton
vector_manager = VectorStoreManager()

