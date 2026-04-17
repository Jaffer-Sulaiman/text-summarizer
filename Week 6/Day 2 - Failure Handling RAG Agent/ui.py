"""
ui.py — Gradio Interface  (Day 2 — Failure Handling)
======================================================
Extends Day 1 UI with:
  - Failure mode badge displayed per response
  - Circuit breaker state visible in System Status tab
  - Retrieval failure and LLM failure shown with distinct icons
  - Manual circuit breaker reset button
"""

import uuid
import requests
import gradio as gr

API_URL = "http://localhost:8000"

CATEGORY_OPTIONS = [
    "general",
    "shipping_policy",
    "tracking",
    "rates_and_zones",
    "customs_and_compliance",
    "claims_and_disputes",
    "faq",
]

FAILURE_MODE_LABELS = {
    "llm_timeout":       "⏱️ LLM Timeout",
    "llm_rate_limited":  "🚦 LLM Rate Limited",
    "llm_auth_error":    "🔑 LLM Auth Error",
    "llm_circuit_open":  "⚡ LLM Circuit Open",
    "llm_unavailable":   "📡 LLM Unavailable",
    "empty_retrieval":   "📭 No Documents Found",
    "retrieval_error":   "🗄️ Knowledge Base Error",
}


# ==============================================================================
# HELPERS
# ==============================================================================

def _post(endpoint, **kwargs):
    try:
        r = requests.post(f"{API_URL}{endpoint}", **kwargs, timeout=120)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"error": r.text}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Cannot connect to API. Make sure api.py is running on port 8000.",
                     "code": "CONNECTION_ERROR"}
    except Exception as e:
        return 500, {"error": str(e)}


def _get(endpoint, **kwargs):
    try:
        r = requests.get(f"{API_URL}{endpoint}", **kwargs, timeout=30)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"error": r.text}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Cannot connect to API."}
    except Exception as e:
        return 500, {"error": str(e)}


def _delete(endpoint):
    try:
        r = requests.delete(f"{API_URL}{endpoint}", timeout=30)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"error": r.text}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Cannot connect to API."}
    except Exception as e:
        return 500, {"error": str(e)}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content) if content is not None else ""


# ==============================================================================
# TAB 1: CHAT
# ==============================================================================

def generate_session_id():
    return str(uuid.uuid4())


def chat_interaction(user_message, chat_history, session_id, category):
    if not user_message or not user_message.strip():
        yield "", chat_history, session_id, "—", "—", "—", "—"
        return

    session_id = session_id.strip() or generate_session_id()
    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "⏳ Processing your request..."},
    ]
    yield "", chat_history, session_id, "⏳", "⏳", "⏳", "⏳"

    payload = {
        "query": user_message,
        "session_id": session_id,
        "category_filter": category if category and category != "general" else None,
    }
    status_code, data = _post("/chat", json=payload)

    if status_code == 200:
        answer          = data.get("answer", "No answer received.")
        sources         = data.get("sources", [])
        needs_escalation= data.get("needs_escalation", False)
        confidence      = data.get("confidence_score", 0.0)
        failure_mode    = data.get("failure_mode")

        chat_history[-1] = {"role": "assistant", "content": answer}
        sources_display  = "\n".join(f"• {s}" for s in sources) if sources else "No sources cited."
        confidence_display = f"{confidence * 100:.0f}%"
        escalation_display = "⚠️ Yes — A human agent will follow up." if needs_escalation else "✅ No"

        # ── NEW: failure mode badge ──
        if failure_mode and failure_mode in FAILURE_MODE_LABELS:
            fm_display = f"**{FAILURE_MODE_LABELS[failure_mode]}**"
        elif failure_mode:
            fm_display = f"**⚠️ {failure_mode}**"
        else:
            fm_display = "✅ None"

    elif status_code == 503:
        # Circuit breaker open or service unavailable
        err = data.get("detail") or data.get("error") or "Service temporarily unavailable."
        if isinstance(err, dict):
            err = err.get("detail") or err.get("error") or str(err)
        chat_history[-1] = {"role": "assistant",
                             "content": f"⚡ **Service Unavailable (503)**: {err}"}
        sources_display = "—"
        confidence_display = "—"
        escalation_display = "—"
        fm_display = "**⚡ Circuit Breaker Open**"

    elif status_code == 504:
        err = data.get("detail") or data.get("error") or "Pipeline timed out."
        if isinstance(err, dict):
            err = err.get("detail") or err.get("error") or str(err)
        chat_history[-1] = {"role": "assistant",
                             "content": f"⏱️ **Timeout (504)**: {err}"}
        sources_display = "—"
        confidence_display = "—"
        escalation_display = "—"
        fm_display = "**⏱️ Pipeline Timeout**"

    else:
        err = data.get("detail") or data.get("error") or str(data)
        if isinstance(err, dict):
            err = err.get("detail") or err.get("error") or str(err)
        chat_history[-1] = {"role": "assistant",
                             "content": f"❌ **Error {status_code}**: {err}"}
        sources_display = "—"
        confidence_display = "—"
        escalation_display = "—"
        fm_display = f"**❌ HTTP {status_code}**"

    yield "", chat_history, session_id, sources_display, confidence_display, escalation_display, fm_display


def clear_session_ui(session_id):
    if session_id:
        _delete(f"/sessions/{session_id}")
    new_id = generate_session_id()
    return [], new_id, "—", "—", "—", "—"


# ==============================================================================
# TAB 2: KNOWLEDGE BASE
# ==============================================================================

def upload_document(file_obj, category):
    if file_obj is None:
        return "⚠️ No file selected."
    try:
        with open(file_obj.name, "rb") as f:
            files = {"file": (file_obj.name.split("\\")[-1].split("/")[-1], f)}
            resp = requests.post(f"{API_URL}/ingest", files=files,
                                 params={"category": category}, timeout=120)

        if resp.status_code in (200, 201):
            data = resp.json()
            if data.get("status") == "duplicate":
                return f"⚠️ **Duplicate**: {data.get('message')}"
            return (
                f"✅ **Ingested Successfully**\n\n"
                f"- **File**: `{data.get('source_name', 'unknown')}`\n"
                f"- **Category**: `{data.get('category', category)}`\n"
                f"- **Chunks Added**: {data.get('chunks_added', 0)}\n"
                f"- **Chunks Discarded**: {data.get('chunks_discarded', 0)}\n"
                f"- **Doc Type**: `{data.get('doc_type_detected', '?')}`\n"
                f"- **Avg Chunk Tokens**: ~{data.get('avg_chunk_tokens', '?')}\n"
                f"- **Uploaded At**: {data.get('upload_ts', '?')}"
            )
        elif resp.status_code == 503:
            err = resp.json()
            detail = err.get("detail") or err.get("error") or "Knowledge base unavailable."
            return f"⚡ **KB Unavailable (503)**: {detail}"
        else:
            err = resp.json()
            detail = err.get("detail") or err.get("error") or str(err)
            if isinstance(detail, dict):
                detail = detail.get("detail") or detail.get("error") or str(detail)
            return f"❌ **Error {resp.status_code}**: {detail}"

    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error**: Make sure api.py is running on port 8000."
    except Exception as e:
        return f"❌ **Unexpected Error**: {str(e)}"


def refresh_documents():
    status_code, data = _get("/kb/documents")
    if status_code != 200:
        return f"❌ Error fetching documents: {data.get('error', 'unknown')}"
    docs = data.get("documents", [])
    if not docs:
        return "📭 Knowledge base is empty. Upload a document to get started."
    lines = [f"### 📚 {len(docs)} Document(s) in Knowledge Base\n"]
    for doc in docs:
        lines.append(
            f"**{doc['source']}**  \n"
            f"  Category: `{doc['category']}`  |  Type: `{doc['file_type']}`  "
            f"|  Pages: {doc['page_count']}  |  Uploaded: {doc['upload_ts']}  \n"
            f"  Hash: `{doc['file_hash']}`\n"
        )
    return "\n---\n".join(lines)


def delete_document_ui(file_hash):
    if not file_hash.strip():
        return "⚠️ Enter a file hash to delete."
    status_code, data = _delete(f"/kb/documents/{file_hash.strip()}")
    if status_code == 200:
        return f"✅ Deleted {data.get('deleted', '?')} chunks for hash `{file_hash}`."
    elif status_code == 404:
        return f"⚠️ No document found with hash `{file_hash}`."
    return f"❌ Error: {data.get('detail') or data.get('error') or str(data)}"


# ==============================================================================
# TAB 3: SYSTEM STATUS
# ==============================================================================

def get_system_status():
    h_code, health  = _get("/health")
    m_code, metrics = _get("/metrics")
    k_code, kb      = _get("/kb/stats")

    lines = ["## 🖥️ System Status\n"]

    if h_code == 200:
        overall = health.get("status", "unknown")
        icon = "🟢" if overall == "ok" else "🟡"
        lines.append(
            f"**API Status**: {icon} {overall.upper()}  \n"
            f"**Version**: {health.get('version', '?')}  \n"
            f"**Uptime**: {health.get('uptime_seconds', 0):.0f}s  \n"
            f"**KB Empty**: {'Yes ⚠️' if health.get('kb_empty') else 'No ✅'}  \n"
            f"**KB Healthy**: {'✅' if health.get('kb_healthy') else '❌'}  \n"
            f"**Active Sessions**: {health.get('active_sessions', 0)}\n"
        )

        # Circuit breaker states (NEW in Day 2)
        breakers = health.get("circuit_breakers", {})
        if breakers:
            lines.append("\n---\n## ⚡ Circuit Breaker States\n")
            for name, b in breakers.items():
                state = b.get("state", "?")
                state_icon = "🟢" if state == "closed" else ("🟡" if state == "half_open" else "🔴")
                lines.append(
                    f"**{name}**: {state_icon} `{state.upper()}`  "
                    f"— Failures: {b.get('failure_count', 0)}/{b.get('failure_threshold', '?')}  "
                    f"— Reset in: {b.get('reset_in_seconds', 0):.0f}s\n"
                )
    else:
        lines.append(f"❌ Health check failed (HTTP {h_code}): {health.get('error', '')}\n")

    lines.append("\n---\n## 📊 Request Metrics\n")
    if m_code == 200:
        lines.append(
            f"- **Total Requests**: {metrics.get('total_requests', 0)}\n"
            f"- **Chat Requests**: {metrics.get('total_chat_requests', 0)}\n"
            f"- **Ingest Requests**: {metrics.get('total_ingest_requests', 0)}\n"
            f"- **Total Errors**: {metrics.get('total_errors', 0)}\n"
            f"- **Pipeline Timeouts**: {metrics.get('total_timeouts', 0)}\n"
            f"- **Circuit Open Hits**: {metrics.get('total_circuit_open', 0)}\n"
            f"- **Avg Latency**: {metrics.get('avg_latency_ms', 0)} ms\n"
        )
    else:
        lines.append(f"❌ Metrics unavailable: {metrics.get('error', '')}\n")

    lines.append("\n---\n## 🗄️ Knowledge Base\n")
    if k_code == 200:
        lines.append(
            f"- **Collection**: `{kb.get('collection_name', '?')}`\n"
            f"- **Total Chunks**: {kb.get('total_chunks', 0)}\n"
            f"- **Total Documents**: {kb.get('total_documents', 0)}\n"
            f"- **Healthy**: {'✅' if kb.get('healthy') else '❌'}\n"
        )
    else:
        lines.append(f"❌ KB stats unavailable: {kb.get('error', '')}\n")

    return "\n".join(lines)


def reset_breakers():
    status_code, data = _post("/breakers/reset")
    if status_code == 200:
        return "✅ All circuit breakers reset to CLOSED state."
    return f"❌ Failed to reset breakers: {data.get('detail') or data.get('error') or str(data)}"


# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================
CSS = """
footer { display: none !important; }
.failure-badge { font-weight: bold; font-size: 0.95em; }
"""

with gr.Blocks(
    title="SwiftShip — Customer Support AI v2",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 📦 SwiftShip Logistics — Customer Support AI  `v2`
        *Gemini · LangGraph · ChromaDB · Full Failure Handling*
        """
    )

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Customer Support Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="SwiftShip Support Agent",
                        height=480,
                        type="messages",
                        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=swiftship"),
                        show_copy_button=True,
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about shipping, tracking, rates, customs, claims...",
                            scale=4,
                            show_label=False,
                        )
                        send_btn = gr.Button("Send 📨", variant="primary", scale=1)
                    clear_btn = gr.Button("🆕 New Session", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Session Settings")
                    session_id_box = gr.Textbox(
                        label="Session ID",
                        value=generate_session_id,
                        info="Auto-generated. Edit to resume a previous session.",
                    )
                    category_dd = gr.Dropdown(
                        label="Knowledge Filter",
                        choices=CATEGORY_OPTIONS,
                        value="general",
                    )

                    gr.Markdown("---\n### 📋 Response Details")
                    sources_box     = gr.Markdown("*Sources appear after first answer.*")
                    confidence_box  = gr.Markdown("**Confidence**: —")
                    escalation_box  = gr.Markdown("**Escalation**: —")
                    # NEW: failure mode badge
                    failure_mode_box = gr.Markdown(
                        "**Failure Mode**: —",
                        elem_classes=["failure-badge"],
                        label="Failure Mode",
                    )

            msg_input.submit(
                chat_interaction,
                inputs=[msg_input, chatbot, session_id_box, category_dd],
                outputs=[msg_input, chatbot, session_id_box,
                         sources_box, confidence_box, escalation_box, failure_mode_box],
            )
            send_btn.click(
                chat_interaction,
                inputs=[msg_input, chatbot, session_id_box, category_dd],
                outputs=[msg_input, chatbot, session_id_box,
                         sources_box, confidence_box, escalation_box, failure_mode_box],
            )
            clear_btn.click(
                clear_session_ui,
                inputs=[session_id_box],
                outputs=[chatbot, session_id_box,
                         sources_box, confidence_box, escalation_box, failure_mode_box],
            )

        # ── Tab 2: Knowledge Base ─────────────────────────────────────────────
        with gr.Tab("📚 Knowledge Base Manager"):
            gr.Markdown(
                "### Upload logistics documents to the SwiftShip knowledge base.\n"
                "Supports **PDF**, **TXT**, **DOCX** · Max **10 MB** · Duplicates auto-detected."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📤 Upload Document")
                    upload_file = gr.File(label="Select File", file_types=[".pdf", ".txt", ".docx"])
                    upload_category = gr.Dropdown(
                        label="Document Category", choices=CATEGORY_OPTIONS, value="general"
                    )
                    upload_btn = gr.Button("⬆️ Upload & Ingest", variant="primary")
                    upload_status = gr.Markdown("*Upload status appears here.*")

                with gr.Column():
                    gr.Markdown("#### 🗑️ Delete Document")
                    delete_hash = gr.Textbox(label="File Hash", placeholder="Paste hash from document list")
                    delete_btn  = gr.Button("🗑️ Delete", variant="stop")
                    delete_status = gr.Markdown("")

            gr.Markdown("---\n#### 📋 Current Documents")
            refresh_btn = gr.Button("🔄 Refresh Document List")
            doc_list_box = gr.Markdown("*Click Refresh to load.*")

            upload_btn.click(upload_document, inputs=[upload_file, upload_category], outputs=[upload_status])
            upload_file.upload(upload_document, inputs=[upload_file, upload_category], outputs=[upload_status])
            refresh_btn.click(refresh_documents, outputs=[doc_list_box])
            delete_btn.click(delete_document_ui, inputs=[delete_hash], outputs=[delete_status])

        # ── Tab 3: System Status ──────────────────────────────────────────────
        with gr.Tab("📊 System Status"):
            gr.Markdown("Live status — fetched on demand. Includes **circuit breaker** states.")
            with gr.Row():
                status_refresh_btn    = gr.Button("🔄 Refresh Status", variant="primary")
                reset_breakers_btn    = gr.Button("⚡ Reset Circuit Breakers", variant="stop")
            reset_status_box = gr.Markdown("")
            status_box       = gr.Markdown("*Click Refresh Status to load.*")

            status_refresh_btn.click(get_system_status, outputs=[status_box])
            reset_breakers_btn.click(reset_breakers, outputs=[reset_status_box])

    gr.Markdown(
        """
        ---
        *SwiftShip Customer Support AI · Week 6 Day 2 · FastAPI + LangGraph + ChromaDB + Gemini*
        """
    )

if __name__ == "__main__":
    demo.queue().launch(server_port=7860, show_error=True, share=False)
