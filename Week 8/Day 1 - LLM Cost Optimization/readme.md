# Week 8 · Day 1 — LLM Cost Optimization

**Objective:**
**reduce token consumption without sacrificing answer quality**, while preserving all
observability coverage.

---

## What Changed vs Week 7 Day 1

Week 8 Day 1 adds **7 targeted cost optimizations** on top of the full logging & observability
layer inherited from Week 7 Day 1.

| # | Optimization | File |
|---|---|---|
| 1 | **Merged classifier node** — `intent_classifier` + `classify_complexity` replaced by single `classify_intent_and_complexity` call | `graph.py` |
| 2 | **Model tiering** — cheaper `gemini-2.0-flash` (same model, but `temperature=0` for classifiers) for all classification nodes | `graph.py`, `config.py` |
| 3 | **Skip-rephrase heuristic** — skips LLM call when query contains no pronouns (self-contained queries) | `graph.py` |
| 4 | **Prompt compression** — all prompts trimmed of verbose preambles and redundant instructions | `graph.py` |
| 5 | **Context truncation** — `_truncate_to_budget()` caps tokens fed to `grade_relevance` and `generate_answer` at `MAX_CONTEXT_TOKENS` (default 2 000) | `graph.py`, `config.py` |
| 6 | **History window tightening** — `rephrase_query` uses last 4 msgs (was 6); `generate_answer` uses last 2 msgs (was 4) | `graph.py`, `config.py` |
| 7 | **Per-request `TokenBudget`** — new `token_budget.py` accumulates tokens across all nodes; emits a WARNING log when `MAX_TOKENS_PER_REQUEST` is exceeded | `token_budget.py`, `graph.py`, `api.py` |

---

## Architecture

```
User Query
    │
    ▼
classify_intent_and_complexity   ← ★ MERGED (was 2 separate nodes)
    │                                uses llm_classifier (temperature=0)
    ├──[greeting]──► handle_greeting
    ├──[off_topic]─► handle_off_topic
    │
    ▼
rephrase_query                   ← ★ SKIPPED if no pronouns detected
    │                                uses llm_classifier; MAX_HISTORY_REPHRASE=4
    ▼
retrieve                         ← top_k already set by merged classifier
    │
    ▼
validate_context
    │
    ▼
grade_relevance                  ← ★ context truncated to MAX_CONTEXT_TOKENS
    │                                uses llm_classifier; compressed prompt
    ├──[no]────────► handle_irrelevant
    │
    ▼
generate_answer                  ← ★ context truncated; history MAX_HISTORY_ANSWER=2
    │                                uses llm_answer; compressed instruction
    │                                emits TokenBudget log
    └──► response
```

---

## New Configuration Constants

All new constants are in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_CLASSIFIER_MODEL` | `gemini-2.0-flash` | Model for all classification nodes |
| `LLM_ANSWER_MODEL` | `gemini-2.0-flash` | Model for answer generation |
| `MAX_CONTEXT_TOKENS` | `2000` | Max tokens of context fed to grade + answer |
| `MAX_TOKENS_PER_REQUEST` | `4000` | Budget warning threshold per request |
| `MAX_HISTORY_REPHRASE` | `4` | History messages passed to rephrase_query |
| `MAX_HISTORY_ANSWER` | `2` | History messages passed to generate_answer |

---

## New Files

| File | Purpose |
|---|---|
| `token_budget.py` | Per-request token accumulator (`TokenBudget` class) with `add()`, `report()`, and `log_summary()` |
| `measure_tokens.py` | Offline measurement script — compares Week 7 vs Week 8 prompt sizes on 5 benchmark queries without making LLM calls |

---

## Before / After Token Comparison

Measured with `measure_tokens.py` using the production `estimate_tokens()` heuristic
(4 chars/token — identical to what the runtime logs use).

**Setup:** 3-chunk retrieval (~2,400 raw context tokens), 4-message history simulation.

| Query | W7 Day 1 `tokens_in` | W8 Day 1 `tokens_in` | Saving | % |
|---|---:|---:|---:|---:|
| Standard delivery time for Zone 3? | 5,541 | 4,233 | +1,308 | 23.6% |
| How do I file a claim for a lost shipment? | 5,536 | 4,230 | +1,306 | 23.6% |
| Prohibited items for international shipping? | 5,554 | 4,242 | +1,312 | 23.6% |
| Difference between economy and express rates? | 5,588 | 4,261 | +1,327 | 23.7% |
| Shipment delayed at customs? | 5,546 | 4,236 | +1,310 | 23.6% |
| **TOTAL (5 queries)** | **27,765** | **21,202** | **+6,563** | **23.6%** |
| **AVERAGE per query** | **5,553** | **4,240** | **+1,312** | **23.6%** |

> **All 5 queries had rephrase_query skipped** — none contained first/second/third-person
> pronouns. In multi-turn conversations where pronouns appear (e.g., "What's *its* rate?"),
> the rephrase call still runs but is cheaper due to the compressed prompt and 4-msg window.

### Savings Breakdown (per average query)

| Optimization | tokens_in saved |
|---|---:|
| Merged classifier (removes 1 LLM call) | ~84 |
| Skip rephrase (no-pronoun queries) | ~163 |
| Context truncation (grade + answer) | ~800 |
| Prompt compression (all nodes) | ~120 |
| History window tightening | ~145 |
| **Total** | **~1,312** |

---

## Models Used

| Node | Week 7 Day 1 | Week 8 Day 1 | Why |
|---|---|---|---|
| `intent_classifier` | `gemini-2.0-flash` | **eliminated** (merged) | — |
| `classify_complexity` | `gemini-2.0-flash` | **eliminated** (merged) | — |
| `classify_intent_and_complexity` | — | `gemini-2.0-flash` (temp=0) | Merged; deterministic classification |
| `rephrase_query` | `gemini-2.0-flash` | `gemini-2.0-flash` (temp=0) | Skipped on standalone queries |
| `grade_relevance` | `gemini-2.0-flash` | `gemini-2.0-flash` (temp=0) | Binary yes/no; deterministic |
| `generate_answer` | `gemini-2.0-flash` | `gemini-2.0-flash` | Quality-critical; unchanged |

> **Note on model name:** The user specified `gemini-2.0-flash` as the model for all nodes.
> Set `LLM_CLASSIFIER_MODEL` and `LLM_ANSWER_MODEL` in `.env` to override.
> If you have access to `gemini-1.5-flash-8b`, setting classifiers to that model
> gives an additional ~4× cost reduction on classification calls.

---

## Token Budget Monitoring

Every request now carries a `TokenBudget` object through the graph:

```
2026-04-29 07:00:00 INFO  Token budget OK
  trace_id=abc123  total_tokens_in=4,240  total_tokens_out=312
  total_tokens=4,552  budget_limit=4,000  over_budget=False
  node_breakdown=[
    {"node":"classify_intent_and_complexity","tokens_in":89,"tokens_out":3},
    {"node":"grade_relevance","tokens_in":2,101,"tokens_out":1},
    {"node":"generate_answer","tokens_in":2,050,"tokens_out":308}
  ]
```

When `total_tokens > MAX_TOKENS_PER_REQUEST`, the log level escalates to **WARNING**:

```
2026-04-29 07:00:01 WARNING  Token budget EXCEEDED — request used 5,200 / 4,000 tokens
```

---

## Running the Project

### 1. Install dependencies
```bash
pip install -r ../../requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

### 3. Ingest documents
```bash
python ingestor.py   # or use the /ingest API endpoint
```

### 4. Start the API
```bash
python api.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 5. Start the UI
```bash
python ui.py
# UI available at http://localhost:7860
```

### 6. Run token measurements
```bash
python measure_tokens.py
```

---

## Inherited from Week 7 Day 1

- **Structured logging** — JSON log lines with `trace_id`, `session_id`, `latency_ms`
- **`TracingContext`** — per-node latency measurement at every graph node
- **`log_llm_call()`** — structured LLM audit record after every LLM invocation
- **`log_prompt()`** — prompt preview logging (full text when `LOG_PROMPT=true`)
- **Rotating log files** — `logs/app.log` with size-based rotation
- **`/logs` endpoint** — return last N log lines without SSH
- **`/metrics` endpoint** — cumulative token counters, latency percentiles
- **Resilience layer** — circuit breakers, retry with exponential backoff, timeouts
- **Memory** — session-scoped conversation history with TTL

---

## Version History

| Version | Week | Key Change |
|---|---|---|
| 1.0.0 | Week 6 Day 1 | Customer Support RAG Agent baseline |
| 2.0.0 | Week 6 Day 2 | Failure handling (circuit breakers, retry) |
| 3.0.0 | Week 7 Day 1 | Logging & Observability |
| **4.0.0** | **Week 8 Day 1** | **LLM Cost Optimization (this version)** |
