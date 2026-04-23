"""
vectorstore.py — ChromaDB Storage Layer  (Week 7 Day 1 — Logging & Observability)
===================================================================================
Extends Day 2 with deep retrieval and ingestion observability:

NEW in Week 7 Day 1:
  - log_retrieval() called after every retrieve() with: query, top_k, category_filter,
    chunks_returned, chunk_sizes, total_context_chars, estimated_context_tokens, latency_ms
  - Ingest logs now include: char_count, avg_chunk_tokens, doc_type, latency_ms (via TimingContext)
  - trace_id accepted by retrieve() and threaded into all log entries
  - health_check() logs the outcome with chunk count for monitoring
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
from observability import log_retrieval, estimate_tokens

log = get_logger("vectorstore")

# ---------------------------------------------------------------------------
# Chunking constants
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
        chunk_count = self._safe_count()
        log.info(
            "VectorStoreManager ready",
            extra={"collection": COLLECTION_NAME, "chunks": chunk_count},
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _safe_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return -1

    def _collection_count(self) -> int:
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

    # ── Public: health check ──────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """
        Probe ChromaDB reachability.
        Returns {\"healthy\": bool, \"chunks\": int, \"error\": str|None}.
        """
        try:
            count = self._collection_count()
            log.info(
                "ChromaDB health check passed",
                extra={"healthy": True, "chunks": count},
            )
            return {"healthy": True, "chunks": count, "error": None}
        except VectorStoreError as e:
            log.error(
                "ChromaDB health check failed",
                extra={"healthy": False, "error": str(e)},
            )
            return {"healthy": False, "chunks": -1, "error": str(e)}
        except Exception as e:
            log.error(
                "ChromaDB health check — unexpected error",
                extra={"healthy": False, "error": str(e)},
            )
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
        trace_id: str = "",
    ) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"status": "error", "message": "No text provided.", "code": "EMPTY_CONTENT"}

        file_hash = self._file_hash(text)

        # Duplicate check
        try:
            existing = self.vectorstore.get(where={"file_hash": file_hash})
            if existing and len(existing["ids"]) > 0:
                log.info(
                    "Duplicate document rejected",
                    extra={
                        "trace_id": trace_id,
                        "source": source_name,
                        "file_hash": file_hash,
                    },
                )
                return {
                    "status": "duplicate",
                    "message": f"'{source_name}' is already in the knowledge base.",
                    "code": "DUPLICATE_DOCUMENT",
                    "chunks_added": 0,
                }
        except Exception as e:
            log.warning(
                "Duplicate check failed — proceeding with ingest",
                extra={"trace_id": trace_id, "error": str(e)},
            )

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
        avg_chunk_chars = int(sum(len(c) for c in chunks) / len(chunks))
        avg_chunk_tokens = int(avg_chunk_chars / 4)
        total_tokens = estimate_tokens(text)

        log.info(
            "Ingest pre-flight",
            extra={
                "trace_id": trace_id,
                "source": source_name,
                "category": category,
                "file_type": file_type,
                "doc_type": doc_type,
                "char_count": len(text),
                "total_tokens_estimate": total_tokens,
                "page_count": page_count,
                "raw_chunks": len(raw_chunks),
                "clean_chunks": len(chunks),
                "discarded_chunks": len(raw_chunks) - len(chunks),
                "avg_chunk_chars": avg_chunk_chars,
                "avg_chunk_tokens": avg_chunk_tokens,
            },
        )

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
            with TimingContext(log, "vectorstore.add_documents",
                               source=source_name, trace_id=trace_id):
                self.vectorstore.add_documents(documents)
        except Exception as e:
            log.error(
                "ChromaDB write failed",
                extra={"trace_id": trace_id, "source": source_name, "error": str(e)},
            )
            raise VectorStoreError(f"Failed to write to ChromaDB: {e}")

        log.info(
            "Document ingested successfully",
            extra={
                "trace_id": trace_id,
                "source": source_name,
                "category": category,
                "chunks_added": len(chunks),
                "chunks_discarded": len(raw_chunks) - len(chunks),
                "avg_chunk_tokens": avg_chunk_tokens,
                "total_tokens_estimate": total_tokens,
                "upload_ts": upload_ts,
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

    # ── Public: retrieve (with timeout + circuit breaker + observability) ──────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_SIMPLE,
        category_filter: Optional[str] = None,
        trace_id: str = "",
    ) -> List[Document]:
        """
        Retrieve top_k most relevant chunks.

        Wrapped with:
          - vectorstore_breaker:  rejects calls when ChromaDB keeps failing
          - invoke_with_timeout:  raises LLMTimeoutError on slow queries
          - log_retrieval():      emits full retrieval audit record
        """
        if self.is_empty():
            log.info(
                "Retrieval skipped — knowledge base is empty",
                extra={"trace_id": trace_id},
            )
            return []

        try:
            count = self._collection_count()
        except VectorStoreError:
            count = top_k

        capped_k = min(top_k, count)
        search_kwargs: Dict[str, Any] = {"k": capped_k}
        if category_filter and category_filter.strip():
            search_kwargs["filter"] = {"category": category_filter}

        log.info(
            "Retrieval starting",
            extra={
                "trace_id": trace_id,
                "query_chars": len(query),
                "query_preview": query[:150].replace("\n", " ↵ "),
                "top_k_requested": top_k,
                "top_k_capped": capped_k,
                "category_filter": category_filter or "none",
                "total_chunks_in_kb": count,
            },
        )

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        def _do_retrieve():
            return retriever.invoke(query)

        start = time.perf_counter()
        success = True
        error_msg = None
        results: List[Document] = []

        try:
            results = vectorstore_breaker.call(
                invoke_with_timeout,
                _do_retrieve,
                timeout_seconds=VECTORSTORE_TIMEOUT_SECONDS,
            )
        except LLMTimeoutError as e:
            success = False
            error_msg = str(e)
            raise VectorStoreError(f"ChromaDB retrieval timed out: {e}")
        except Exception as e:
            success = False
            error_msg = str(e)
            raise VectorStoreError(f"ChromaDB retrieval failed: {e}")
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            chunk_sizes = [len(doc.page_content) for doc in results]
            log_retrieval(
                log,
                trace_id=trace_id,
                query=query,
                top_k=capped_k,
                category_filter=category_filter,
                chunks_returned=len(results),
                latency_ms=latency_ms,
                chunk_sizes=chunk_sizes if chunk_sizes else None,
                success=success,
                error=error_msg,
            )

        return results

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
            log.info(
                "Document deleted",
                extra={"file_hash": file_hash, "chunks_deleted": len(ids)},
            )
            return {"status": "success", "deleted": len(ids)}
        except Exception as e:
            log.error("Delete failed", extra={"file_hash": file_hash, "error": str(e)})
            return {"status": "error", "deleted": 0, "message": str(e)}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
vector_manager = VectorStoreManager()
