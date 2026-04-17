"""
vectorstore.py — ChromaDB Storage Layer
========================================
Manages document ingestion, retrieval, listing, and deletion
against a persistent ChromaDB collection.

Key capabilities vs Week 5:
  - Rich metadata per chunk: source, file_hash, category, doc_type,
    upload_ts (ISO-8601), page_count
  - Category-aware retrieval: optional `category_filter` narrows search
    to a specific sub-collection of documents
  - `list_documents()` — returns one entry per source file (deduplicated)
  - `delete_document(file_hash)` — removes all chunks for one document
  - `get_stats()` — collection-level observability info
  - Adaptive chunking (dense vs prose) inherited and hardened from Week 5
"""

import os
import re
import hashlib
import time
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K_SIMPLE,
    TOP_K_COMPLEX,
)
from logger import get_logger, TimingContext

log = get_logger("vectorstore")

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------
MIN_CHUNK_CHARS = 60
DENSE_CHUNK_SIZE = 512
PROSE_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
PROSE_SENTENCE_THRESHOLD = 18   # avg words/sentence above this → prose mode

# Signal quality threshold: at least 40% of chars must be alphabetic
ALPHA_RATIO_THRESHOLD = 0.40


class VectorStoreManager:
    """
    Singleton-friendly manager for the ChromaDB logistics knowledge base.
    """

    def __init__(self):
        log.info("Initialising VectorStoreManager", extra={"db_dir": CHROMA_DB_DIR})

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

        self._separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        log.info(
            "VectorStoreManager ready",
            extra={"collection": COLLECTION_NAME, "chunks": self._collection_count()},
        )

    # ------------------------------------------------------------------
    # Private: Adaptive chunking
    # ------------------------------------------------------------------
    def _estimate_doc_type(self, text: str) -> str:
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
        """Quality filter: discard too-short or mostly non-alphabetic chunks."""
        clean = []
        for chunk in chunks:
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            alpha_ratio = sum(c.isalpha() for c in chunk) / max(len(chunk), 1)
            if alpha_ratio < ALPHA_RATIO_THRESHOLD:
                continue
            clean.append(chunk)
        return clean

    # ------------------------------------------------------------------
    # Private: Collection helpers
    # ------------------------------------------------------------------
    def _collection_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0

    def _file_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public: Status checks
    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        return self._collection_count() == 0

    def get_stats(self) -> Dict[str, Any]:
        """Return collection-level stats for the /kb/stats endpoint."""
        count = self._collection_count()
        docs = self.list_documents()
        return {
            "total_chunks": count,
            "total_documents": len(docs),
            "collection_name": COLLECTION_NAME,
            "db_dir": CHROMA_DB_DIR,
            "documents": docs,
        }

    # ------------------------------------------------------------------
    # Public: Ingest
    # ------------------------------------------------------------------
    def ingest_document(
        self,
        text: str,
        source_name: str,
        category: str = "general",
        page_count: int = 1,
        file_type: str = "txt",
    ) -> Dict[str, Any]:
        """
        Chunk and add a document to ChromaDB.

        Args:
            text:        Extracted plain text.
            source_name: Original filename (shown in citations).
            category:    Logistics domain tag for filtered retrieval.
            page_count:  Number of pages/sections (for metadata).
            file_type:   "pdf" | "txt" | "docx".

        Returns:
            Status dict consumed by the API response model.
        """
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided.", "code": "EMPTY_CONTENT"}

        file_hash = self._file_hash(text)

        # Duplicate check —— query by metadata filter
        try:
            existing = self.vectorstore.get(where={"file_hash": file_hash})
            if existing and len(existing["ids"]) > 0:
                log.info("Duplicate document rejected", extra={"source": source_name})
                return {
                    "status": "duplicate",
                    "message": f"'{source_name}' is already in the knowledge base.",
                    "code": "DUPLICATE_DOCUMENT",
                    "chunks_added": 0,
                }
        except Exception as e:
            log.warning("Duplicate check failed — proceeding", extra={"error": str(e)})

        # Adaptive chunking
        doc_type = self._estimate_doc_type(text)
        splitter = self._make_splitter(doc_type)
        raw_chunks = splitter.split_text(text)
        chunks = self._filter_chunks(raw_chunks)

        if not chunks:
            return {
                "status": "error",
                "message": "No usable chunks extracted. Document may be blank or non-textual.",
                "code": "EMPTY_CONTENT",
            }

        upload_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        avg_chunk_tokens = int(sum(len(c) for c in chunks) / len(chunks) / 4)

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "file_hash": file_hash,
                    "category": category,
                    "doc_type": doc_type,
                    "file_type": file_type,
                    "page_count": page_count,
                    "upload_ts": upload_ts,
                },
            )
            for chunk in chunks
        ]

        with TimingContext(log, "vectorstore.add_documents", source=source_name):
            self.vectorstore.add_documents(documents)

        log.info(
            "Document ingested",
            extra={
                "source": source_name,
                "category": category,
                "doc_type": doc_type,
                "chunks_added": len(chunks),
                "chunks_discarded": len(raw_chunks) - len(chunks),
            },
        )

        return {
            "status": "success",
            "message": f"Successfully ingested '{source_name}'.",
            "source_name": source_name,
            "file_hash": file_hash,
            "category": category,
            "chunks_added": len(chunks),
            "chunks_discarded": len(raw_chunks) - len(chunks),
            "doc_type_detected": doc_type,
            "avg_chunk_tokens": avg_chunk_tokens,
            "upload_ts": upload_ts,
        }

    # ------------------------------------------------------------------
    # Public: Retrieve
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_SIMPLE,
        category_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve the top_k most relevant chunks.

        Args:
            query:           User's (possibly rephrased) question.
            top_k:           Number of chunks to retrieve.
            category_filter: If set, restrict search to chunks with this category.

        Returns:
            List of LangChain Document objects with page_content and metadata.
        """
        if self.is_empty():
            return []

        capped_k = min(top_k, self._collection_count())

        search_kwargs: Dict[str, Any] = {"k": capped_k}
        if category_filter and category_filter.strip():
            search_kwargs["filter"] = {"category": category_filter}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        with TimingContext(log, "vectorstore.retrieve", top_k=capped_k, category=category_filter):
            results = retriever.invoke(query)

        return results

    # ------------------------------------------------------------------
    # Public: List documents
    # ------------------------------------------------------------------
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return one metadata entry per unique source document.
        Deduplicates by file_hash so chunked documents appear as one record.
        """
        try:
            data = self.vectorstore.get(include=["metadatas"])
        except Exception as e:
            log.error("Failed to list documents", extra={"error": str(e)})
            return []

        seen_hashes: set = set()
        docs: List[Dict[str, Any]] = []

        for meta in (data.get("metadatas") or []):
            fh = meta.get("file_hash", "")
            if fh and fh not in seen_hashes:
                seen_hashes.add(fh)
                docs.append({
                    "source": meta.get("source", "unknown"),
                    "file_hash": fh,
                    "category": meta.get("category", "general"),
                    "doc_type": meta.get("doc_type", "unknown"),
                    "file_type": meta.get("file_type", "unknown"),
                    "page_count": meta.get("page_count", 0),
                    "upload_ts": meta.get("upload_ts", ""),
                })

        return docs

    # ------------------------------------------------------------------
    # Public: Delete document
    # ------------------------------------------------------------------
    def delete_document(self, file_hash: str) -> Dict[str, Any]:
        """
        Remove all chunks belonging to a document identified by its file_hash.
        Returns {"deleted": N, "status": "success"|"not_found"}.
        """
        try:
            existing = self.vectorstore.get(where={"file_hash": file_hash})
            ids_to_delete = existing.get("ids", [])
            if not ids_to_delete:
                return {"status": "not_found", "deleted": 0}
            self.vectorstore.delete(ids=ids_to_delete)
            log.info(
                "Document deleted",
                extra={"file_hash": file_hash, "chunks_deleted": len(ids_to_delete)},
            )
            return {"status": "success", "deleted": len(ids_to_delete)}
        except Exception as e:
            log.error("Delete failed", extra={"file_hash": file_hash, "error": str(e)})
            return {"status": "error", "deleted": 0, "message": str(e)}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
vector_manager = VectorStoreManager()
