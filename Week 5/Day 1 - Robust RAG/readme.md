# Week 5 Day 1 - Robust RAG 

## Features Accomplished

> [!TIP]
> **Industry Standards Adhered To**
> We maintained strict separation of concerns (Data, Logic, API, and UI) just like production enterprise applications.

1. **Persistent Vector Database (`vectorstore.py`)**:
   - Configured `ChromaDB` to save locally to a hidden `.chroma_db` folder within the project.
   - Successfully integrated `HuggingFaceEmbeddings` containing an open-source, highly efficient embedding model (`all-MiniLM-L6-v2`) via `langchain-huggingface`.
   - Prevented duplicates using MD5 string hashing on the backend.

2. **Advanced Agentic Logic (`graph.py`)**:
   - Constructed a LangGraph state machine incorporating conversational history.
   - Addressed Hallucinations: We've attached a strict "Relevance Grader". The system automatically evaluates if the chunks pulled from ChromaDB actually match the user's question before generation begins, falling back gracefully if not.

3. **FastAPI Backend (`api.py`)**
   - Implemented a hardened `/upload` endpoint supporting PDF ingestion (via `pypdf`) and straightforward `.txt` chunks.
   - Implemented a conversational `/chat` endpoint backed by comprehensive Pydantic validation schemas.
   - Added DB status guardrails to proactively stop queries on an empty database.

4. **Gradio Frontend (`ui.py`)**
   - Separated the User Interface into two logical tabs: **Knowledge Base Manager** for document upload/indexing validation, and **RAG Chat** for user interactions.

## Verification

The core files (`api.py`, `ui.py`, `graph.py`, `vectorstore.py`) have been statically verified for syntax and logic correctness. All dependencies—including `chromadb`, `sentence-transformers`, `python-multipart`, and `pypdf`—were automatically appended to the centralized `requirements.txt`.

> [!NOTE]
> **How to Run This Mini-Project**
> Execute these commands from the root of your `text_summarizer` project workspace:
> 
> Terminal 1 (Backend):
> ```bash
> .\venv\Scripts\activate
> python "Week 5\Day 1 - Robust RAG\api.py"
> ```
> 
> Terminal 2 (Frontend):
> ```bash
> .\venv\Scripts\activate
> python "Week 5\Day 1 - Robust RAG\ui.py"
> ```
