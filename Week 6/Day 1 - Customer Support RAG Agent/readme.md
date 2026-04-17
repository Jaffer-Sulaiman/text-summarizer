# 📦 Week 6 · Day 1 — Customer Support AI RAG Chat Agent (Logistics)

> **Domain**: SwiftShip Logistics Customer Support  
> **Stack**: FastAPI · ChromaDB · Google Gemini · LangGraph · Gradio  
> **Theme**: Production-ready demo with fundamentally solid, scalable architecture

---

## 🏗️ Project Structure

```
Week 6/Day 1 - Customer Support RAG Agent/
│
├── config.py           ← Centralized env + constants (Layer 1)
├── logger.py           ← Structured JSON logging (Layer 2)
├── ingestor.py         ← Document parsing: PDF / TXT / DOCX (Layer 3)
├── vectorstore.py      ← ChromaDB with metadata & category filters (Layer 4)
├── memory.py           ← Session-keyed conversational memory (Layer 5)
├── graph.py            ← LangGraph 10-node pipeline (Layer 6)
├── api.py              ← FastAPI layer + middleware + 8 endpoints (Layer 7)
├── ui.py               ← Gradio 3-tab branded interface (Layer 8)
│
└── sample_docs/
    ├── shipping_policy.txt     ← Demo: Zones, SLAs, rates, prohibited items
    └── tracking_guide.txt      ← Demo: Status codes, delay handling, claims
```

---

## ▶️ How to Run

```powershell
# From the root of text_summarizer/
.\\venv\\Scripts\\activate

# Terminal 1 — Start the Backend API
python "Week 6\\Day 1 - Customer Support RAG Agent\\api.py"

# Terminal 2 — Start the Gradio UI
python "Week 6\\Day 1 - Customer Support RAG Agent\\ui.py"
```

| Service | URL |
|---|---|
| Gradio UI | http://localhost:7860 |
| FastAPI Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| ReDoc | http://localhost:8000/redoc |

---

## ✅ Quick Demo Steps

1. Open the UI at http://localhost:7860
2. Go to **📚 Knowledge Base Manager** tab
3. Upload `sample_docs/shipping_policy.txt` → select category `shipping_policy`
4. Upload `sample_docs/tracking_guide.txt` → select category `tracking`
5. Go to **💬 Customer Support Chat** tab
6. Try these queries:
   - `"What are the shipping zones and their distance ranges?"`
   - `"How many delivery attempts does SwiftShip make?"`
   - `"Hello!"` → greeting handler
   - `"What is the weather today?"` → off-topic guardrail
   - After asking about zones: `"What are the rates for zone 3?"` → memory + rephrasing

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness + readiness probe |
| `GET` | `/metrics` | Request stats + avg latency |
| `POST` | `/ingest` | Upload document to knowledge base |
| `GET` | `/kb/documents` | List all ingested documents |
| `GET` | `/kb/stats` | ChromaDB collection stats |
| `DELETE` | `/kb/documents/{hash}` | Delete document by MD5 hash |
| `POST` | `/chat` | Main RAG chat endpoint |
| `DELETE` | `/sessions/{id}` | Clear conversation session |
| `GET` | `/sessions` | List active sessions |

### Chat Request / Response Schema

**POST /chat**
```json
{
  "query": "What are the shipping zones?",
  "session_id": "uuid-here",
  "category_filter": "shipping_policy"
}
```

**Response**
```json
{
  "answer": "SwiftShip operates across 6 domestic zones...",
  "session_id": "uuid-here",
  "sources": ["shipping_policy.txt"],
  "needs_escalation": false,
  "confidence_score": 0.92
}
```

---

## 🏛️ Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    %% ─────────────── USER LAYER ───────────────
    USER(["👤 Customer / Support Agent"])

    subgraph UI["🖥️ Layer 8 — Gradio UI (ui.py)"]
        TAB1["💬 Chat Tab\n(session_id, category filter,\nsource citations, escalation banner)"]
        TAB2["📚 KB Manager Tab\n(upload, category tag,\ndelete, list docs)"]
        TAB3["📊 Status Tab\n(health, metrics, KB stats)"]
    end

    %% ─────────────── API LAYER ───────────────
    subgraph API["⚡ Layer 7 — FastAPI API Layer (api.py)"]
        MW1["CORSMiddleware"]
        MW2["RateLimitMiddleware\n(60 req/min/IP)"]
        MW3["RequestTimingMiddleware\n(structured log every request)"]
        AUTH["API Key Auth\n(X-API-Key, optional)"]
        EP_CHAT["POST /chat"]
        EP_INGEST["POST /ingest"]
        EP_KB["GET /kb/documents\nGET /kb/stats\nDELETE /kb/documents/{hash}"]
        EP_SYS["GET /health\nGET /metrics\nGET /sessions"]
    end

    %% ─────────────── INGESTOR LAYER ───────────────
    subgraph ING["📄 Layer 3 — Ingestor (ingestor.py)"]
        EXT_PDF["PDF Parser\n(pypdf)"]
        EXT_TXT["TXT Parser\n(utf-8 fallback)"]
        EXT_DOCX["DOCX Parser\n(python-docx)"]
        VAL_FILE["Validations:\n• Extension allowlist\n• Max 10 MB\n• Non-empty content"]
    end

    %% ─────────────── VECTOR STORE LAYER ───────────────
    subgraph VS["🗄️ Layer 4 — VectorStore (vectorstore.py)"]
        EMBED["HuggingFace Embeddings\n(all-MiniLM-L6-v2)"]
        CHUNK["Adaptive Chunker\n(dense 512 / prose 1500)"]
        DEDUP["Deduplication\n(MD5 hash check)"]
        CHROMA[("ChromaDB\n(persistent on disk)")]
        META["Rich Metadata:\nsource · file_hash · category\ndoc_type · upload_ts · page_count"]
        CAT_FILTER["Category Filter\n(optional WHERE clause)"]
    end

    %% ─────────────── MEMORY LAYER ───────────────
    subgraph MEM["🧠 Layer 5 — Session Memory (memory.py)"]
        STORE["In-Memory Dict\n(session_id → messages)"]
        LOCK["threading.Lock\n(concurrent access safe)"]
        TTL["TTL Eviction\n(2 hrs idle → purge)"]
        WINDOW["Sliding Window\n(max 20 turns)"]
    end

    %% ─────────────── LANGGRAPH PIPELINE ───────────────
    subgraph GRAPH["🔄 Layer 6 — LangGraph Pipeline (graph.py)"]
        direction TB
        N1["1️⃣ intent_classifier\n(logistics / greeting / off_topic)"]
        N2["2️⃣ rephrase_query\n(standalone reformulation with history)"]
        N3["3️⃣ classify_complexity\n(simple k=3 / complex k=6)"]
        N4["4️⃣ retrieve\n(ChromaDB similarity + category filter)"]
        N5["5️⃣ validate_context\n(junk scrub · redaction · signal density)"]
        N6["6️⃣ grade_relevance\n(LLM-based context grader)"]
        N7["7️⃣ generate_answer\n(grounded · cited · confidence scored)"]

        T_GREET["handle_greeting"]
        T_OT["handle_off_topic"]
        T_DEG["handle_degraded"]
        T_IRR["handle_irrelevant"]
        T_ESC["8️⃣ escalate_to_human\n(appends escalation notice)"]

        N1 -->|"greeting"| T_GREET
        N1 -->|"off_topic"| T_OT
        N1 -->|"logistics"| N2
        N2 --> N3 --> N4 --> N5
        N5 -->|"all bad"| T_DEG
        N5 -->|"ok"| N6
        N6 -->|"no"| T_IRR
        N6 -->|"yes"| N7
        N7 -->|"needs_escalation=true"| T_ESC
        N7 -->|"confident"| END_NODE(["✅ END"])
        T_GREET --> END_NODE
        T_OT --> END_NODE
        T_DEG --> END_NODE
        T_IRR --> END_NODE
        T_ESC --> END_NODE
    end

    %% ─────────────── SUPPORT LAYERS ───────────────
    subgraph CFG["⚙️ Layer 1 — Config (config.py)"]
        ENV[".env\n(GOOGLE_API_KEY, LLM_MODEL,\nRATE_LIMIT_RPM, SESSION_TTL...)"]
    end

    subgraph LOG["📝 Layer 2 — Logger (logger.py)"]
        JLOG["JSON Structured Log\n(ts · level · component · latency_ms)"]
        TIMING["TimingContext\n(per-operation latency instrumentation)"]
    end

    subgraph LLM_SVC["🤖 Gemini LLM (Google GenAI)"]
        GEMINI["gemini-1.5-flash\n(intent · complexity · grade · answer)"]
    end

    %% ─────────────── CONNECTIONS ───────────────
    USER <--> TAB1 & TAB2 & TAB3
    TAB1 <-->|"POST /chat"| EP_CHAT
    TAB2 <-->|"POST /ingest"| EP_INGEST
    TAB2 <-->|"GET /kb/*"| EP_KB
    TAB3 <-->|"GET /health /metrics"| EP_SYS

    EP_INGEST --> VAL_FILE --> ING
    ING --> CHUNK --> DEDUP --> EMBED --> META --> CHROMA

    EP_CHAT --> MEM
    MEM --> GRAPH
    GRAPH --> CHROMA
    GRAPH --> LLM_SVC
    GRAPH --> MEM

    CFG --> API & GRAPH & VS & MEM & ING
    LOG --> API & GRAPH & VS & ING
```

---

## 🆚 What's New vs Week 5

| Feature | Week 5 (Day 1 & 2) | Week 6 Day 1 |
|---|---|---|
| **Domain** | Generic Q&A | Logistics customer support |
| **Session memory** | Client-side history array | Server-side `session_id` store with TTL |
| **Intent guard** | ❌ | ✅ (logistics / greeting / off_topic) |
| **Query reform** | ❌ | ✅ (standalone rephrasing using history) |
| **Escalation** | ❌ | ✅ (confidence threshold + escalation node) |
| **File formats** | PDF + TXT | PDF + TXT + DOCX |
| **Metadata** | source + hash | + category + doc_type + upload_ts + page_count |
| **Category filter** | ❌ | ✅ (retrieval narrowed by topic tag) |
| **Auth** | ❌ | ✅ (X-API-Key, optional) |
| **Rate limiting** | ❌ | ✅ (60 req/min/IP sliding window) |
| **Structured logging** | `print()` | JSON structured logger with TimingContext |
| **API endpoints** | 2 (`/upload`, `/chat`) | 9 endpoints incl. `/health`, `/metrics`, `/kb/*` |
| **Error schema** | Raw HTTPException | Standardized `ErrorResponse` with error codes |
| **Graph nodes** | 7 nodes | 10 nodes (+ intent, rephrase, escalate) |
| **UI tabs** | 2 tabs | 3 tabs + confidence badge + sources panel |

---

## 🔑 Error Codes

| Code | HTTP | Meaning |
|---|---|---|
| `INVALID_FILE_TYPE` | 422 | Unsupported extension |
| `FILE_TOO_LARGE` | 422 | Exceeds 10 MB limit |
| `EMPTY_CONTENT` | 422 | No text extractable |
| `DUPLICATE_DOCUMENT` | 200 | Already in knowledge base |
| `INVALID_CATEGORY` | 422 | Unknown category label |
| `EMPTY_QUERY` | 422 | Blank chat message |
| `GRAPH_EXECUTION_ERROR` | 500 | LangGraph pipeline failure |
| `SESSION_NOT_FOUND` | 404 | Invalid session ID for DELETE |
| `DOCUMENT_NOT_FOUND` | 404 | Invalid hash for DELETE |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INVALID_API_KEY` | 401 | Bad/missing X-API-Key header |
