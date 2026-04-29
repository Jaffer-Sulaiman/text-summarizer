"""
token_budget.py — Per-Request Token Budget Tracker  (Week 8 Day 1 — LLM Cost Optimization)
============================================================================================
Tracks cumulative token spend across every LLM node within a single request.

Usage (in graph.py):
    from token_budget import TokenBudget
    budget = TokenBudget(trace_id=tid)
    budget.add("intent_classifier", tokens_in=80, tokens_out=2)
    budget.add("generate_answer",   tokens_in=500, tokens_out=120)
    report = budget.report()   # dict with per-node breakdown + totals
    budget.log_summary(log)    # emit structured log; warns if over budget

Design decisions:
  - Pure Python dataclass — no external dependencies
  - Thread-safe via a simple list accumulator (one budget per request, no sharing)
  - Budget warning threshold read from config (MAX_TOKENS_PER_REQUEST)
  - report() always returns valid data even if add() was never called
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from config import MAX_TOKENS_PER_REQUEST


@dataclass
class _NodeEntry:
    node: str
    tokens_in: int
    tokens_out: int

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out


class TokenBudget:
    """
    Accumulates token counts for every LLM call in a single request.

    Parameters
    ----------
    trace_id : str
        The UUID4 trace ID for the current request (used in log output).
    max_tokens : int
        Budget warning threshold (defaults to MAX_TOKENS_PER_REQUEST from config).
    """

    def __init__(self, trace_id: str = "", max_tokens: int = MAX_TOKENS_PER_REQUEST):
        self._trace_id = trace_id
        self._max_tokens = max_tokens
        self._entries: List[_NodeEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, node: str, tokens_in: int, tokens_out: int = 0) -> None:
        """Record token usage for one LLM call."""
        self._entries.append(_NodeEntry(node=node, tokens_in=tokens_in, tokens_out=tokens_out))

    def report(self) -> Dict:
        """
        Return a structured report of token spend.

        Returns
        -------
        dict with keys:
          - trace_id
          - total_tokens_in
          - total_tokens_out
          - total_tokens
          - budget_limit
          - over_budget (bool)
          - nodes: list of {node, tokens_in, tokens_out, total}
        """
        total_in  = sum(e.tokens_in  for e in self._entries)
        total_out = sum(e.tokens_out for e in self._entries)
        total     = total_in + total_out
        return {
            "trace_id":        self._trace_id,
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_tokens":    total,
            "budget_limit":    self._max_tokens,
            "over_budget":     total > self._max_tokens,
            "nodes": [
                {
                    "node":       e.node,
                    "tokens_in":  e.tokens_in,
                    "tokens_out": e.tokens_out,
                    "total":      e.total,
                }
                for e in self._entries
            ],
        }

    def log_summary(self, log: logging.Logger) -> None:
        """
        Emit a structured log entry with the full token budget report.
        Logs at WARNING level if the request is over budget, INFO otherwise.
        """
        rpt = self.report()
        level = logging.WARNING if rpt["over_budget"] else logging.INFO
        msg   = (
            "Token budget EXCEEDED — request used "
            f"{rpt['total_tokens']:,} / {self._max_tokens:,} tokens"
            if rpt["over_budget"]
            else "Token budget OK"
        )
        log.log(
            level,
            msg,
            extra={
                "trace_id":         self._trace_id,
                "total_tokens_in":  rpt["total_tokens_in"],
                "total_tokens_out": rpt["total_tokens_out"],
                "total_tokens":     rpt["total_tokens"],
                "budget_limit":     self._max_tokens,
                "over_budget":      rpt["over_budget"],
                "node_breakdown":   rpt["nodes"],
            },
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        """Quick access to grand total without a full report() call."""
        return sum(e.total for e in self._entries)
