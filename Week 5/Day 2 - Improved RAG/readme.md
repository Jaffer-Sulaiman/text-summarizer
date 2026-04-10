# 📈 Week 5 · Day 2 — Improved RAG Pipeline

> **Evaluation Metric: Answer Accuracy**
> Every improvement in this version was designed to directly increase the factual correctness of the LLM's responses.

---

## What's New vs Day 1

### 1. 🧩 Adaptive Chunk Sizing (`vectorstore.py`)

| | Day 1 | Day 2 |
|---|---|---|
| Strategy | Fixed `chunk_size = 1000` for all documents | Heuristic detects document style first |
| Dense docs (transcripts, logs) | Over-packs unrelated topics → noisy chunks | Uses `chunk_size = 512` to keep ideas tight |
| Prose docs (reports, papers) | Splits mid-sentence → broken context | Uses `chunk_size = 1500` to preserve full sentences |
| Separators | Included `.` (splits "Dr. Smith") | Uses `. ` / `! ` / `? ` to split only at sentence ends |
| Junk filter | None — headings/whitespace stored as chunks | Discards any chunk where <40% of chars are alphabetic |
| Minimum length | None | Discards chunks shorter than 50 characters |
| Observability | Returns `chunks_added` only | Also returns `doc_type_detected`, `chunks_discarded`, `avg_chunk_tokens` |

**Heuristic**: The first 3000 characters of the document are sampled. Average words-per-sentence is computed. If < 18 words/sentence → `dense` mode. Otherwise → `prose` mode.

---

### 2. 🎯 Dynamic Retrieval Count (`graph.py`)

| | Day 1 | Day 2 |
|---|---|---|
| Strategy | Fixed `top_k = 4` always | Classified per query before retrieval |
| Simple queries | Retrieves 4 chunks, most unrelated → confuses LLM | Retrieves exactly 3 tight, high-precision chunks |
| Complex queries | 4 chunks too few for multi-entity answers | Retrieves 6 chunks for broader evidence coverage |
| Small databases | Could request more chunks than exist → ChromaDB error | `min(classified_k, db_collection_count)` cap applied |

**New LangGraph Node**: `classify_query` runs before `retrieve`. Uses a structured LLM call to label the query `simple` or `complex`, then sets `top_k` automatically.

---

### 3. 🧹 Context Quality Validation (`graph.py`)

> **New in Day 2 — was not present in Day 1 at all.**

A new `validate_context` node runs on every retrieved chunk **before** the relevance grader sees it.

| Problem | Day 1 | Day 2 |
|---|---|---|
| Encoding junk (`\x00`, `\ufffd`) from PDF extraction | Passed raw to LLM | Stripped by regex scrubber |
| Separator lines (`----`, `####`, `. . . .`) | Stored and retrieved | Discarded by signal-density check (<40% alpha) |
| Redacted content (`[REDACTED]`, `████`, `***`) | LLM tries to answer anyway → **hallucinates** | Replaced with annotation: `[Note: redacted content …]` |
| All chunks degraded | LLM receives pure noise | Short-circuits to `handle_degraded_context` node → clean message |

---

### 4. 🗣️ Hardened Prompt Format (`graph.py`)

| | Day 1 | Day 2 |
|---|---|---|
| Role declaration | Vague: "helpful and polite AI assistant" | Precise: "precise document Q&A assistant" |
| Context injection | Raw `"\n\n".join(documents)` — anonymous blob | Numbered `[Source 1]`, `[Source 2]` labeled blocks |
| Citation instruction | None | LLM instructed to cite `[Source N]` in Evidence field |
| Output format | Free-form paragraph | Structured: `**Answer**: … ` / `**Evidence**: …` |
| Reasoning instruction | None | Chain-of-thought: "First, silently identify relevant sources…" |
| System role | Used `{"role": "system"}` → corrupted by Gemini's `convert_system_message_to_human` | Removed system role; instructions embedded in first `HumanMessage` |

---

### 5. 🔀 Expanded LangGraph — New Nodes & Routes

```
Day 1 Flow:
  START → retrieve → grade_documents → [generate | handle_irrelevant] → END

Day 2 Flow:
  START → classify_query → retrieve → validate_context
        → [grade_documents | handle_degraded_context]
        → [generate | handle_irrelevant] → END
```

Two new nodes: `classify_query` and `validate_context`.
One new terminal node: `handle_degraded_context`.

---

## How to Run

```powershell
# From the root of text_summarizer/
.\venv\Scripts\activate

# Terminal 1 — Backend
python "Week 5\Day 2 - Improved RAG\api.py"

# Terminal 2 — Frontend
python "Week 5\Day 2 - Improved RAG\ui.py"
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
UI: [http://localhost:7860](http://localhost:7860)
