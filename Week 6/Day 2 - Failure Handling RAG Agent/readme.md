# 📦 Week 6 · Day 2 — Customer Support RAG Agent: Failure Handling

> **Focus**: Every external I/O boundary is hardened against real-world failure
> **Stack**: FastAPI · ChromaDB · Google Gemini · LangGraph · Gradio

---

## 🆚 What's New vs Day 1

| Layer | Day 1 | Day 2 |
|---|---|---|
| **Resilience module** | ❌ | ✅ `resilience.py` — timeout, retry, circuit breaker |
| **LLM timeout** | Hung forever on slow call | 30s timeout → `LLMTimeoutError` |
| **LLM retry** | Single attempt | 2 retries + exponential back-off (1s, 2s) |
| **LLM circuit breaker** | ❌ | ✅ Opens after 5 consecutive LLM failures |
| **ChromaDB circuit breaker** | ❌ | ✅ Opens after 5 consecutive DB failures |
| **Chat pipeline circuit breaker** | ❌ | ✅ API-level; returns HTTP 503 when open |
| **Graph pipeline timeout** | Blocked indefinitely | `asyncio.wait_for` → HTTP 504 |
| **Empty retrieval handling** | Fell through to irrelevant | Dedicated `handle_empty_retrieval` node |
| **Retrieval failure handling** | Unhandled exception | Dedicated `handle_retrieval_failure` node |
| **LLM failure routing** | Generic fallback string | Dedicated `handle_llm_failure` with typed messages |
| **ChromaDB health check** | ❌ | ✅ `health_check()` probed by `/health` |
| **Deep `/health` endpoint** | Liveness only | Probes ChromaDB + all circuit breaker states |
| **`/breakers` endpoint** | ❌ | ✅ Live circuit breaker status |
| **`POST /breakers/reset`** | ❌ | ✅ Manual reset (post-incident recovery) |
| **`failure_mode` in response** | ❌ | ✅ Surfaced in `ChatResponse` and UI badge |
| **Graph nodes** | 12 nodes | 15 nodes (+3 failure nodes) |
| **HTTP status codes** | 200/422/500 | + 503 (circuit open / DB down) / 504 (timeout) |

---

## 🗂️ Project Structure

```
Week 6/Day 2 - Failure Handling RAG Agent/
├── config.py         ← + LLM_TIMEOUT, LLM_MAX_RETRIES, GRAPH_TIMEOUT, CB thresholds
├── logger.py         ← Unchanged from Day 1
├── resilience.py     ← NEW — timeout wrapper, retry w/ backoff, CircuitBreaker
├── ingestor.py       ← Unchanged from Day 1
├── vectorstore.py    ← + health_check(), circuit-breaker-guarded retrieve()
├── memory.py         ← Unchanged from Day 1
├── graph.py          ← + failure_mode state, 3 new failure nodes, typed routing
├── api.py            ← + asyncio timeout, chat circuit breaker, /breakers endpoints
├── ui.py             ← + failure mode badge, circuit breaker status, reset button
└── sample_docs/
    ├── shipping_policy.txt
    └── tracking_guide.txt
```

---

## ▶️ How to Run

```powershell
.\\venv\\Scripts\\activate

# Terminal 1 — Backend API
python "Week 6\\Day 2 - Failure Handling RAG Agent\\api.py"

# Terminal 2 — Gradio UI
python "Week 6\\Day 2 - Failure Handling RAG Agent\\ui.py"
```

| Service | URL |
|---|---|
| Gradio UI | http://localhost:7860 |
| Swagger Docs | http://localhost:8000/docs |
| Health (deep) | http://localhost:8000/health |
| Circuit Breakers | http://localhost:8000/breakers |

---

## 🧪 Failure Scenarios & How They Are Handled

| Scenario | What Happens | User Sees |
|---|---|---|
| **No documents uploaded** | `is_empty()` → graph routes to `handle_empty_retrieval` | "No documents found" + contact info |
| **0 results from ChromaDB** | `retrieve()` returns `[]` → `failure_mode = "empty_retrieval"` | "No documents matched" + suggestions |
| **ChromaDB connection failure** | `VectorStoreError` raised → `failure_mode = "retrieval_error"` | "KB temporarily unavailable" + contact |
| **ChromaDB timeout (>10s)** | `invoke_with_timeout` raises `LLMTimeoutError` → wrapped as `VectorStoreError` | Same as above |
| **ChromaDB repeated failures** | `vectorstore_breaker` opens → rejects further calls | Same as above |
| **LLM call timeout (>30s)** | `LLMTimeoutError` → `failure_mode = "llm_timeout"` | "AI taking too long, try again" |
| **LLM rate limit (429)** | `LLMRateLimitError` → retry ×2 → `failure_mode = "llm_rate_limited"` | "AI at capacity, wait 1 min" |
| **Invalid API key** | `LLMAuthError` → NO retry → `failure_mode = "llm_auth_error"` | "Config issue, team notified" |
| **Network down to Google** | `LLMUnavailableError` → retry ×2 → `failure_mode = "llm_unavailable"` | "AI unreachable, try shortly" |
| **LLM circuit open** | `CircuitOpenError` → `failure_mode = "llm_circuit_open"` | "AI paused, retry in 60s" |
| **Structured output parse fail** | Falls back to plain `llm.invoke` call | Normal answer, lower confidence |
| **Plain fallback also fails** | Hard failure → `failure_mode` set | LLM failure message |
| **Full pipeline timeout (>90s)** | `asyncio.wait_for` raises `TimeoutError` → HTTP 504 | "Request timed out" |
| **Repeated graph crashes** | `chat_circuit_breaker` opens → HTTP 503 | "Service paused, retry in 60s" |
| **Intent classifier LLM error** | Default to `"logistics"` (soft fail) | Pipeline continues normally |
| **Rephrase query LLM error** | Use original question (soft fail) | Pipeline continues normally |
| **Complexity classifier error** | Default to `"simple"` (soft fail) | Pipeline continues normally |
| **Relevance grader error** | Default to `"yes"` (optimistic soft fail) | Pipeline continues normally |
| **Corrupted/junk document chunks** | Validate context strips them (Day 1 logic) | "Degraded context" message |
| **File too large** | `IngestorError` → HTTP 422 + `FILE_TOO_LARGE` | Error in UI upload panel |
| **Bad file type** | `IngestorError` → HTTP 422 + `INVALID_FILE_TYPE` | Error in UI upload panel |

---

## 🏛️ Architecture Diagram (Mermaid)

```mermaid
flowchart TD

    USER(["👤 Customer"])

    subgraph UI["🖥️ Gradio UI  •  ui.py"]
        CHAT["💬 Chat Tab\nfailure mode badge\ncircuit breaker status"]
        KB["📚 KB Manager"]
        STATUS["📊 Status + Reset Breakers"]
    end

    subgraph API["⚡ FastAPI  •  api.py"]
        MW["Middleware\nCORS · Rate Limit · Timing"]
        EP_CHAT["POST /chat\nasyncio.wait_for timeout\nchat circuit breaker"]
        EP_INGEST["POST /ingest"]
        EP_HEALTH["GET /health\nDeep probe: ChromaDB + breakers"]
        EP_BREAKERS["GET /breakers\nPOST /breakers/reset"]
    end

    subgraph RES["🛡️ resilience.py  NEW"]
        TIMEOUT["invoke_with_timeout\n30s LLM · 10s DB"]
        RETRY["invoke_with_retry\n2 retries · exp backoff"]
        CB_LLM["CircuitBreaker\ngemini_llm\n5 failures → OPEN"]
        CB_DB["CircuitBreaker\nchromadb\n5 failures → OPEN"]
        CB_CHAT["CircuitBreaker\nchat_pipeline\n5 failures → 503"]
        EXC["Exception Hierarchy\nLLMTimeout · RateLimit\nAuthError · Unavail\nVectorStoreError"]
    end

    subgraph VS["🗄️ vectorstore.py"]
        HC["health_check()\nprobes ChromaDB count"]
        RET_GUARDED["retrieve()\ncircuit breaker + timeout"]
        CHROMA[("ChromaDB")]
    end

    subgraph GRAPH["🔄 LangGraph Pipeline  •  graph.py   15 nodes"]
        direction TB
        N1["1. intent_classifier\nSOFT FAIL → logistics"]
        N2["2. rephrase_query\nSOFT FAIL → original q"]
        N3["3. classify_complexity\nSOFT FAIL → simple"]
        N4["4. retrieve\nHARD FAIL node"]
        N5["5. validate_context\ninternal only"]
        N6["6. grade_relevance\nSOFT FAIL → yes"]
        N7["7. generate_answer\nHARD FAIL node\nstructured + plain fallback"]

        T_GREET(["handle_greeting"])
        T_OT(["handle_off_topic"])
        T_DEG(["handle_degraded"])
        T_IRR(["handle_irrelevant"])
        T_ESC["escalate_to_human"]

        T_LLM["handle_llm_failure  NEW\nllm_timeout · rate_limited\nauth_error · circuit_open\nunavailable"]
        T_EMPTY["handle_empty_retrieval  NEW\n0 results from ChromaDB"]
        T_RETERR["handle_retrieval_failure  NEW\nChromaDB unavailable"]

        GEND(["END"])

        N1 -->|"greeting"| T_GREET
        N1 -->|"off_topic"| T_OT
        N1 -->|"logistics"| N2
        N2 --> N3 --> N4

        N4 -->|"failure_mode=empty_retrieval"| T_EMPTY
        N4 -->|"failure_mode=retrieval_error"| T_RETERR
        N4 -->|"ok"| N5

        N5 -->|"all bad"| T_DEG
        N5 -->|"ok"| N6
        N6 -->|"no"| T_IRR
        N6 -->|"yes"| N7

        N7 -->|"failure_mode set"| T_LLM
        N7 -->|"needs_escalation"| T_ESC
        N7 -->|"confident"| GEND

        T_GREET --> GEND
        T_OT --> GEND
        T_DEG --> GEND
        T_IRR --> GEND
        T_ESC --> GEND
        T_LLM --> GEND
        T_EMPTY --> GEND
        T_RETERR --> GEND
    end

    subgraph LLM["🤖 Gemini LLM"]
        GEMINI["gemini-1.5-flash\nTimeout 30s · Retry 2x · CB 5 fails"]
    end

    USER <--> CHAT
    USER <--> KB
    USER <--> STATUS

    CHAT <-->|"POST /chat"| EP_CHAT
    KB   <-->|"POST /ingest"| EP_INGEST
    STATUS <-->|"GET /health /breakers"| EP_HEALTH
    STATUS <-->|"POST /breakers/reset"| EP_BREAKERS

    EP_CHAT --> CB_CHAT
    CB_CHAT --> GRAPH

    GRAPH --> RET_GUARDED
    RET_GUARDED --> CB_DB
    CB_DB --> TIMEOUT
    TIMEOUT --> CHROMA

    GRAPH --> CB_LLM
    CB_LLM --> RETRY
    RETRY --> TIMEOUT
    TIMEOUT --> GEMINI

    EP_HEALTH --> HC
    HC --> CHROMA

    EXC -.->|"typed errors"| GRAPH
    CB_LLM -.->|"status"| EP_HEALTH
    CB_DB  -.->|"status"| EP_HEALTH
    CB_CHAT -.->|"status"| EP_HEALTH
```

---

## 🔑 New Error Codes (Day 2 additions)

| Code | HTTP | Trigger |
|---|---|---|
| `LLM_TIMEOUT` | 200 (in answer) | LLM call > 30s |
| `LLM_RATE_LIMITED` | 200 (in answer) | HTTP 429 from Gemini |
| `LLM_AUTH_ERROR` | 200 (in answer) | Bad API key |
| `LLM_CIRCUIT_OPEN` | 200 (in answer) | LLM circuit breaker OPEN |
| `LLM_UNAVAILABLE` | 200 (in answer) | Network error after retries |
| `VECTOR_STORE_ERROR` | 503 | ChromaDB unavailable on ingest |
| `CHAT_CIRCUIT_OPEN` | 503 | Chat pipeline circuit breaker OPEN |
| `GRAPH_TIMEOUT` | 504 | Full pipeline > 90s |
