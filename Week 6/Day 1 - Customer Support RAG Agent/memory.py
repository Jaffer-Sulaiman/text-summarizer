"""
memory.py — Session Conversational Memory Layer
================================================
Thread-safe, session-keyed in-memory store for conversational history.

Design decisions:
  - Keyed by session_id (UUID string) so multiple users can chat
    concurrently without state collision.
  - Sliding window: only the last SESSION_MAX_TURNS messages are kept,
    preventing context window overflow for long conversations.
  - TTL-based eviction: sessions idle for SESSION_TTL_SECONDS are
    auto-pruned on every write to prevent unbounded memory growth.
  - Uses threading.Lock for safe concurrent access under uvicorn workers.

Public interface:
    get_history(session_id)         → List[BaseMessage]
    add_turn(session_id, q, a)      → None
    clear_session(session_id)       → bool
    list_sessions()                 → List[dict]
    session_exists(session_id)      → bool
"""

import time
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import SESSION_TTL_SECONDS, SESSION_MAX_TURNS
from logger import get_logger

log = get_logger("memory")


# ---------------------------------------------------------------------------
# Internal session record
# ---------------------------------------------------------------------------
@dataclass
class _Session:
    messages: List[BaseMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Session Memory Store
# ---------------------------------------------------------------------------
class SessionMemoryStore:
    """
    Thread-safe in-memory store for multi-session conversational history.
    One instance is created at module level and imported as a singleton.
    """

    def __init__(self):
        self._store: Dict[str, _Session] = {}
        self._lock = threading.Lock()
        log.info("SessionMemoryStore initialized")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evict_expired(self) -> None:
        """Remove sessions that have been idle longer than SESSION_TTL_SECONDS.
        Called on every write — no background thread required."""
        now = time.time()
        expired = [
            sid for sid, s in self._store.items()
            if now - s.last_active > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._store[sid]
            log.info("Session evicted (TTL)", extra={"session_id": sid})

    def _get_or_create(self, session_id: str) -> _Session:
        """Return existing session or create a new one (must be called under lock)."""
        if session_id not in self._store:
            self._store[session_id] = _Session()
            log.info("New session created", extra={"session_id": session_id})
        return self._store[session_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """
        Return the conversation history for a session.
        Returns an empty list if the session does not exist yet
        (first message scenario).
        """
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return []
            return list(session.messages)

    def add_turn(self, session_id: str, human_text: str, ai_text: str) -> None:
        """
        Append a human+AI exchange to the session.
        Applies sliding-window truncation after adding.
        """
        with self._lock:
            self._evict_expired()
            session = self._get_or_create(session_id)

            session.messages.append(HumanMessage(content=human_text))
            session.messages.append(AIMessage(content=ai_text))
            session.turn_count += 1
            session.last_active = time.time()

            # Sliding window: keep only the last N messages
            max_msgs = SESSION_MAX_TURNS * 2  # each turn = 2 messages
            if len(session.messages) > max_msgs:
                session.messages = session.messages[-max_msgs:]
                log.info(
                    "Session window trimmed",
                    extra={"session_id": session_id, "kept": max_msgs},
                )

    def clear_session(self, session_id: str) -> bool:
        """
        Delete a session. Returns True if it existed, False otherwise.
        """
        with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                log.info("Session cleared", extra={"session_id": session_id})
                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return metadata snapshot for all active sessions."""
        with self._lock:
            now = time.time()
            return [
                {
                    "session_id": sid,
                    "turn_count": s.turn_count,
                    "message_count": len(s.messages),
                    "age_seconds": round(now - s.created_at),
                    "idle_seconds": round(now - s.last_active),
                }
                for sid, s in self._store.items()
            ]

    def active_count(self) -> int:
        """Return count of currently active sessions."""
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
memory_store = SessionMemoryStore()
