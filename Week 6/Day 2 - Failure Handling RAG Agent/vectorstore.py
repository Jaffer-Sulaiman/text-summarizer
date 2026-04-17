"""
vectorstore.py — ChromaDB Storage Layer  (Day 2 — Failure Handling)
=====================================================================
Extends Day 1 with:
  - health_check()         — used by /health endpoint to probe ChromaDB
  - Circuit breaker guard  — via vectorstore_breaker from resilience.py
  - VectorStoreError       — raised on any ChromaDB failure so graph nodes
                             can catch and route to handle_retrieval_failure
  - Timeout on retrieve()  — wraps the blocking Chroma call in a thread
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
    VECTORSTORE_TIMEOUT_SECONDS,
)
from resilience import (
    VectorStoreError,
    vectorstore_breaker,
    invoke_with_timeout,
    LLMTimeoutError,
)
from logger import get_logger, TimingContext

log = get_logger("vectorstore")

# ---------------------------------------------------------------------------
# Chunking constants (same as Day 1)
# ---------------------------------------------------------------------------
MIN_CHUNK_CHARS = 60
DENSE_CHUNK_SIZE = 512
PROSE_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
PROSE_SENTENCE_THRESHOLD = 18
ALPHA_RATIO_THRESHOLD = 0.40


class VectorStoreManager:

    def __init__(self):
        log.info("Initialising VectorStoreManager", extra={"db_dir": CHROMA_DB_DIR})
        try:
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
        except Exception as e:
            log.error("VectorStoreManager init failed", extra={"error": str(e)})
            raise VectorStoreError(f"ChromaDB initialisation failed: {e}")

        self._separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        log.info(
            "VectorStoreManager ready",
            extra={"collection": COLLECTION_NAME, "chunks": self._safe_count()},
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _safe_count(self) -> int:
        """Count without raising — used in __init__ logging."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return -1

    def _collection_count(self) -> int:
        """Count with VectorStoreError on failure."""
        try:
            return self.vectorstore._collection.count()
        except Exception as e:
            raise VectorStoreError(f"ChromaDB count failed: {e}")

    def _file_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

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
        clean = []
        for chunk in chunks:
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            alpha_ratio = sum(c.isalpha() for c in chunk) / max(len(chunk), 1)
            if alpha_ratio < ALPHA_RATIO_THRESHOLD:
                continue
            clean.append(chunk)
        return clean

    # ── Public: health check (NEW in Day 2) ───────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """
        Probe ChromaDB reachability.
        Returns {"healthy": bool, "chunks": int, "error": str|None}.
        Used by the /health endpoint for deep readiness checks.
        """
        try:
            count = self._collection_count()
            return {"healthy": True, "chunks": count, "error": None}
        except VectorStoreError as e:
            return {"healthy": False, "chunks": -1, "error": str(e)}
        except Exception as e:
            return {"healthy": False, "chunks": -1, "error": f"Unexpected: {e}"}

    # ── Public: status helpers ────────────────────────────────────────────────

    def is_empty(self) -> bool:
        try:
            return self._collection_count() == 0
        except VectorStoreError:
            return True

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self._collection_count()
        except VectorStoreError as e:
            return {"error": str(e), "healthy": False}
        docs = self.list_documents()
        return {
            "total_chunks": count,
            "total_documents": len(docs),
            "collection_name": COLLECTION_NAME,
            "db_dir": CHROMA_DB_DIR,
            "documents": docs,
            "healthy": True,
        }

    # ── Public: ingest ────────────────────────────────────────────────────────

    def ingest_document(
        self,
        text: str,
        source_name: str,
        category: str = "general",
        page_count: int = 1,
        file_type: str = "txt",
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided.", "code": "EMPTY_CONTENT"}

        file_hash = self._file_hash(text)

        # Duplicate check
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

        doc_type = self._estimate_doc_type(text)
        splitter = self._make_splitter(doc_type)
        raw_chunks = splitter.split_text(text)
        chunks = self._filter_chunks(raw_chunks)

        if not chunks:
            return {
                "status": "error",
                "message": "No usable chunks extracted.",
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

        try:
            with TimingContext(log, "vectorstore.add_documents", source=source_name):
                self.vectorstore.add_documents(documents)
        except Exception as e:
            log.error("ChromaDB write failed", extra={"source": source_name, "error": str(e)})
            raise VectorStoreError(f"Failed to write to ChromaDB: {e}")

        log.info(
            "Document ingested",
            extra={
                "source": source_name,
                "category": category,
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

    # ── Public: retrieve (with timeout + circuit breaker) ─────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_SIMPLE,
        category_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve top_k most relevant chunks.

        Wrapped with:
          - vectorstore_breaker:  rejects calls when ChromaDB keeps failing
          - invoke_with_timeout:  raises LLMTimeoutError → treated as
                                  retrieval_timeout in the graph node
          - VectorStoreError:     raised on ChromaDB exceptions
        """
        if self.is_empty():
            return []

        try:
            count = self._collection_count()
        except VectorStoreError:
            count = top_k  # Can't verify — proceed optimistically

        capped_k = min(top_k, count)
        search_kwargs: Dict[str, Any] = {"k": capped_k}
        if category_filter and category_filter.strip():
            search_kwargs["filter"] = {"category": category_filter}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        def _do_retrieve():
            return retriever.invoke(query)

        try:
            with TimingContext(log, "vectorstore.retrieve", top_k=capped_k):
                results = vectorstore_breaker.call(
                    invoke_with_timeout,
                    _do_retrieve,
                    timeout_seconds=VECTORSTORE_TIMEOUT_SECONDS,
                )
            return results
        except LLMTimeoutError as e:
            # Re-raise as VectorStoreError so graph sees one error type
            raise VectorStoreError(f"ChromaDB retrieval timed out: {e}")
        except Exception as e:
            raise VectorStoreError(f"ChromaDB retrieval failed: {e}")

    # ── Public: list documents ────────────────────────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        try:
            data = self.vectorstore.get(include=["metadatas"])
        except Exception as e:
            log.error("Failed to list documents", extra={"error": str(e)})
            return []

        seen: set = set()
        docs: List[Dict[str, Any]] = []
        for meta in (data.get("metadatas") or []):
            fh = meta.get("file_hash", "")
            if fh and fh not in seen:
                seen.add(fh)
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

    # ── Public: delete ────────────────────────────────────────────────────────

    def delete_document(self, file_hash: str) -> Dict[str, Any]:
        try:
            existing = self.vectorstore.get(where={"file_hash": file_hash})
            ids = existing.get("ids", [])
            if not ids:
                return {"status": "not_found", "deleted": 0}
            self.vectorstore.delete(ids=ids)
            log.info("Document deleted", extra={"file_hash": file_hash, "chunks": len(ids)})
            return {"status": "success", "deleted": len(ids)}
        except Exception as e:
            log.error("Delete failed", extra={"file_hash": file_hash, "error": str(e)})
            return {"status": "error", "deleted": 0, "message": str(e)}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
vector_manager = VectorStoreManager()
