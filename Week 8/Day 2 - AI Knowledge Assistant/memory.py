"""
memory.py — Session Conversational Memory Layer  (Week 7 Day 1)
================================================================
Extends Day 2 with richer session lifecycle observability:

NEW in Week 7 Day 1:
  - add_turn() logs: turn_count, session age, idle time
  - TTL proximity warning when session is within 10% of expiry
  - get_history() logs session hit/miss for trace correlation
  - list_sessions() logs snapshot count for monitoring
"""

import time
import threading
from typing import List, Dict, Any
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import SESSION_TTL_SECONDS, SESSION_MAX_TURNS
from logger import get_logger

log = get_logger("memory")

# TTL proximity warning threshold — warn when idle time exceeds this fraction of TTL
_TTL_WARNING_FRACTION = 0.90


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
        log.info("SessionMemoryStore initialized", extra={"ttl_seconds": SESSION_TTL_SECONDS,
                                                          "max_turns": SESSION_MAX_TURNS})

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
            s = self._store[sid]
            log.info(
                "Session evicted — TTL expired",
                extra={
                    "session_id": sid,
                    "turn_count": s.turn_count,
                    "age_seconds": round(now - s.created_at),
                    "idle_seconds": round(now - s.last_active),
                    "ttl_seconds": SESSION_TTL_SECONDS,
                },
            )
            del self._store[sid]

    def _get_or_create(self, session_id: str) -> _Session:
        """Return existing session or create a new one (must be called under lock)."""
        if session_id not in self._store:
            self._store[session_id] = _Session()
            log.info(
                "New session created",
                extra={"session_id": session_id, "active_sessions": len(self._store)},
            )
        return self._store[session_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def get_history(self, session_id: str, trace_id: str = "") -> List[BaseMessage]:
        """
        Return the conversation history for a session.
        Returns an empty list if the session does not exist yet
        (first message scenario).
        """
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                log.debug(
                    "Session history miss — new session",
                    extra={"session_id": session_id, "trace_id": trace_id},
                )
                return []
            log.debug(
                "Session history retrieved",
                extra={
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "message_count": len(session.messages),
                    "turn_count": session.turn_count,
                },
            )
            return list(session.messages)

    def add_turn(
        self,
        session_id: str,
        human_text: str,
        ai_text: str,
        trace_id: str = "",
    ) -> None:
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
            max_msgs = SESSION_MAX_TURNS * 2
            trimmed = False
            if len(session.messages) > max_msgs:
                session.messages = session.messages[-max_msgs:]
                trimmed = True
                log.info(
                    "Session window trimmed",
                    extra={
                        "session_id": session_id,
                        "trace_id": trace_id,
                        "kept_messages": max_msgs,
                    },
                )

            now = time.time()
            idle_fraction = (now - session.last_active) / SESSION_TTL_SECONDS

            # Log turn summary
            log.info(
                "Session turn recorded",
                extra={
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "turn_count": session.turn_count,
                    "message_count": len(session.messages),
                    "age_seconds": round(now - session.created_at),
                    "idle_seconds": round(now - session.last_active),
                    "trimmed": trimmed,
                    "human_chars": len(human_text),
                    "ai_chars": len(ai_text),
                },
            )

            # TTL proximity warning — alert when session is >90% through its TTL
            idle_seconds = now - session.last_active
            if idle_seconds > SESSION_TTL_SECONDS * _TTL_WARNING_FRACTION:
                log.warning(
                    "Session approaching TTL expiry",
                    extra={
                        "session_id": session_id,
                        "idle_seconds": round(idle_seconds),
                        "ttl_seconds": SESSION_TTL_SECONDS,
                        "expires_in_seconds": round(SESSION_TTL_SECONDS - idle_seconds),
                    },
                )

    def clear_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed, False otherwise."""
        with self._lock:
            if session_id in self._store:
                s = self._store[session_id]
                now = time.time()
                log.info(
                    "Session cleared by request",
                    extra={
                        "session_id": session_id,
                        "turn_count": s.turn_count,
                        "age_seconds": round(now - s.created_at),
                    },
                )
                del self._store[session_id]
                return True
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return metadata snapshot for all active sessions."""
        with self._lock:
            now = time.time()
            sessions = [
                {
                    "session_id": sid,
                    "turn_count": s.turn_count,
                    "message_count": len(s.messages),
                    "age_seconds": round(now - s.created_at),
                    "idle_seconds": round(now - s.last_active),
                }
                for sid, s in self._store.items()
            ]
        log.debug(
            "Sessions listed",
            extra={"active_sessions": len(sessions)},
        )
        return sessions

    def active_count(self) -> int:
        """Return count of currently active sessions."""
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
memory_store = SessionMemoryStore()
