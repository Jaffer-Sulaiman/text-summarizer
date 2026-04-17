"""
ui.py — Gradio Logistics-Branded Interface
============================================
3-tab Gradio UI for the SwiftShip Customer Support RAG Agent:

  Tab 1: 💬 Customer Support Chat
    - Auto-generated session ID (editable)
    - Category filter dropdown for targeted retrieval
    - Conversational chatbot with streaming placeholder
    - Source citations panel
    - Confidence score + escalation banner
    - "New Session" button to reset

  Tab 2: 📚 Knowledge Base Manager
    - File upload with category tagging
    - Ingestion status with rich metadata
    - Live document list (refreshable)
    - Document deletion by hash

  Tab 3: 📊 System Status
    - Health check results
    - KB stats (chunks, documents, collection name)
    - Active session count
    - API metrics (request counts, avg latency)
"""

import uuid
import json
import requests
import gradio as gr

# ---------------------------------------------------------------------------
# Backend connection
# ---------------------------------------------------------------------------
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


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _post(endpoint: str, **kwargs) -> tuple[int, dict]:
    """Thin wrapper for POST requests with connection error handling."""
    try:
        resp = requests.post(f"{API_URL}{endpoint}", **kwargs, timeout=120)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {"error": resp.text}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Cannot connect to API", "code": "CONNECTION_ERROR",
                     "detail": "Make sure api.py is running on port 8000."}
    except Exception as e:
        return 500, {"error": str(e), "code": "CLIENT_ERROR"}


def _get(endpoint: str, **kwargs) -> tuple[int, dict]:
    """Thin wrapper for GET requests."""
    try:
        resp = requests.get(f"{API_URL}{endpoint}", **kwargs, timeout=30)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {"error": resp.text}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Cannot connect to API. Make sure api.py is running."}
    except Exception as e:
        return 500, {"error": str(e)}


def _delete(endpoint: str) -> tuple[int, dict]:
    try:
        resp = requests.delete(f"{API_URL}{endpoint}", timeout=30)
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {"error": resp.text}
    except requests.exceptions.ConnectionError:
        return 503, {"error": "Cannot connect to API."}
    except Exception as e:
        return 500, {"error": str(e)}


def _extract_text(content) -> str:
    """Normalize Gradio message content to plain string."""
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
# TAB 1: CHAT LOGIC
# ==============================================================================

def generate_session_id() -> str:
    return str(uuid.uuid4())


def chat_interaction(user_message: str, chat_history: list, session_id: str, category: str):
    """Handle one chat turn: send to /chat, update history, extract metadata."""
    if not user_message or not user_message.strip():
        yield "", chat_history, session_id, "—", "—", "—"
        return

    session_id = session_id.strip() or generate_session_id()

    # Add user message and loading placeholder
    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "⏳ Thinking..."},
    ]
    yield "", chat_history, session_id, "⏳", "⏳", "⏳"

    # Call API
    payload = {
        "query": user_message,
        "session_id": session_id,
        "category_filter": category if category and category != "general" else None,
    }
    status_code, data = _post("/chat", json=payload)

    if status_code == 200:
        answer = data.get("answer", "No answer received.")
        sources = data.get("sources", [])
        needs_escalation = data.get("needs_escalation", False)
        confidence = data.get("confidence_score", 0.0)

        chat_history[-1] = {"role": "assistant", "content": answer}

        sources_display = "\n".join(f"• {s}" for s in sources) if sources else "No sources cited."
        confidence_display = f"{confidence * 100:.0f}%"
        escalation_display = "⚠️ Yes — A human agent will follow up." if needs_escalation else "✅ No"

    else:
        error = data.get("detail") or data.get("error") or str(data)
        if isinstance(error, dict):
            error = error.get("detail") or error.get("error") or str(error)
        chat_history[-1] = {"role": "assistant", "content": f"❌ **Error {status_code}**: {error}"}
        sources_display = "—"
        confidence_display = "—"
        escalation_display = "—"

    yield "", chat_history, session_id, sources_display, confidence_display, escalation_display


def clear_session_ui(session_id: str):
    """Clear backend session and reset UI."""
    if session_id:
        _delete(f"/sessions/{session_id}")
    new_id = generate_session_id()
    return [], new_id, "—", "—", "—"


# ==============================================================================
# TAB 2: KNOWLEDGE BASE LOGIC
# ==============================================================================

def upload_document(file_obj, category: str):
    """Upload a document to the knowledge base."""
    if file_obj is None:
        return "⚠️ No file selected."

    try:
        with open(file_obj.name, "rb") as f:
            files = {"file": (file_obj.name.split("\\")[-1].split("/")[-1], f)}
            params = {"category": category}
            resp = requests.post(f"{API_URL}/ingest", files=files, params=params, timeout=120)

        if resp.status_code in (200, 201):
            data = resp.json()
            status = data.get("status", "")
            if status == "duplicate":
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
    """Fetch and format document list from the API."""
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


def delete_document_ui(file_hash: str):
    """Delete a document by hash."""
    if not file_hash.strip():
        return "⚠️ Enter a file hash to delete."
    status_code, data = _delete(f"/kb/documents/{file_hash.strip()}")
    if status_code == 200:
        return f"✅ Deleted {data.get('deleted', '?')} chunks for hash `{file_hash}`."
    elif status_code == 404:
        return f"⚠️ No document found with hash `{file_hash}`."
    else:
        return f"❌ Error: {data.get('detail') or data.get('error') or str(data)}"


# ==============================================================================
# TAB 3: SYSTEM STATUS LOGIC
# ==============================================================================

def get_system_status():
    """Fetch and format health + metrics + KB stats."""
    # Health
    h_code, health = _get("/health")
    # Metrics
    m_code, metrics = _get("/metrics")
    # KB Stats
    k_code, kb = _get("/kb/stats")

    lines = ["## 🖥️ System Status\n"]

    if h_code == 200:
        status_icon = "🟢" if health.get("status") == "ok" else "🔴"
        lines.append(
            f"**API Status**: {status_icon} {health.get('status', 'unknown').upper()}  \n"
            f"**Version**: {health.get('version', '?')}  \n"
            f"**Uptime**: {health.get('uptime_seconds', 0):.0f}s  \n"
            f"**KB Empty**: {'Yes ⚠️' if health.get('kb_empty') else 'No ✅'}  \n"
            f"**Active Sessions**: {health.get('active_sessions', 0)}\n"
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
            f"- **Avg Latency**: {metrics.get('avg_latency_ms', 0)} ms\n"
            f"- **Active Sessions**: {metrics.get('active_sessions', 0)}\n"
        )
    else:
        lines.append(f"❌ Metrics unavailable: {metrics.get('error', '')}\n")

    lines.append("\n---\n## 🗄️ Knowledge Base Stats\n")
    if k_code == 200:
        lines.append(
            f"- **Collection**: `{kb.get('collection_name', '?')}`\n"
            f"- **Total Chunks**: {kb.get('total_chunks', 0)}\n"
            f"- **Total Documents**: {kb.get('total_documents', 0)}\n"
            f"- **DB Directory**: `{kb.get('db_dir', '?')}`\n"
        )
    else:
        lines.append(f"❌ KB stats unavailable: {kb.get('error', '')}\n")

    return "\n".join(lines)


# ==============================================================================
# GRADIO INTERFACE
# ==============================================================================
CSS = """
.escalation-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 8px; border-radius: 4px; }
.confidence-box { font-size: 1.2em; font-weight: bold; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="SwiftShip — Customer Support AI",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CSS,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        # 📦 SwiftShip Logistics — Customer Support AI
        *Powered by Gemini · LangGraph · ChromaDB*
        """
    )

    with gr.Tabs():

        # =====================================================================
        # TAB 1: CHAT
        # =====================================================================
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
                            lines=1,
                        )
                        send_btn = gr.Button("Send 📨", variant="primary", scale=1)

                    with gr.Row():
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
                        info="Restricts retrieval to this document category.",
                    )

                    gr.Markdown("---\n### 📋 Response Details")
                    sources_box = gr.Markdown(
                        value="*Sources will appear here after the first answer.*",
                        label="Sources",
                    )
                    confidence_box = gr.Markdown(
                        value="**Confidence**: —",
                        label="Confidence",
                        elem_classes=["confidence-box"],
                    )
                    escalation_box = gr.Markdown(
                        value="**Escalation**: —",
                        label="Escalation",
                    )

            # Events
            msg_input.submit(
                chat_interaction,
                inputs=[msg_input, chatbot, session_id_box, category_dd],
                outputs=[msg_input, chatbot, session_id_box, sources_box, confidence_box, escalation_box],
            )
            send_btn.click(
                chat_interaction,
                inputs=[msg_input, chatbot, session_id_box, category_dd],
                outputs=[msg_input, chatbot, session_id_box, sources_box, confidence_box, escalation_box],
            )
            clear_btn.click(
                clear_session_ui,
                inputs=[session_id_box],
                outputs=[chatbot, session_id_box, sources_box, confidence_box, escalation_box],
            )

        # =====================================================================
        # TAB 2: KNOWLEDGE BASE
        # =====================================================================
        with gr.Tab("📚 Knowledge Base Manager"):
            gr.Markdown(
                "### Upload documents to the SwiftShip knowledge base.\n"
                "Supports **PDF**, **TXT**, and **DOCX**. Max file size: **10 MB**. "
                "Duplicate documents are automatically detected."
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📤 Upload Document")
                    upload_file = gr.File(
                        label="Select File",
                        file_types=[".pdf", ".txt", ".docx"],
                    )
                    upload_category = gr.Dropdown(
                        label="Document Category",
                        choices=CATEGORY_OPTIONS,
                        value="general",
                        info="Tag this document so users can filter by topic.",
                    )
                    upload_btn = gr.Button("⬆️ Upload & Ingest", variant="primary")
                    upload_status = gr.Markdown("*Upload status will appear here.*")

                with gr.Column():
                    gr.Markdown("#### 🗑️ Delete Document")
                    delete_hash_input = gr.Textbox(
                        label="File Hash",
                        placeholder="Paste the file hash from the document list below",
                    )
                    delete_btn = gr.Button("🗑️ Delete Document", variant="stop")
                    delete_status = gr.Markdown("")

            gr.Markdown("---\n#### 📋 Current Documents")
            refresh_btn = gr.Button("🔄 Refresh Document List")
            doc_list_box = gr.Markdown("*Click Refresh to load documents.*")

            # Events
            upload_btn.click(upload_document, inputs=[upload_file, upload_category], outputs=[upload_status])
            upload_file.upload(upload_document, inputs=[upload_file, upload_category], outputs=[upload_status])
            refresh_btn.click(refresh_documents, outputs=[doc_list_box])
            delete_btn.click(delete_document_ui, inputs=[delete_hash_input], outputs=[delete_status])

        # =====================================================================
        # TAB 3: SYSTEM STATUS
        # =====================================================================
        with gr.Tab("📊 System Status"):
            gr.Markdown("Live status dashboard — fetches current data from the API on demand.")
            status_refresh_btn = gr.Button("🔄 Refresh Status", variant="primary")
            status_box = gr.Markdown("*Click Refresh Status to load.*")

            status_refresh_btn.click(get_system_status, outputs=[status_box])

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
        ---
        *SwiftShip Customer Support AI · Week 6 Day 1 · Built with FastAPI + LangGraph + ChromaDB + Gemini*
        """
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    demo.queue().launch(
        server_port=7860,
        show_error=True,
        share=False,
    )
