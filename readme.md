# 📄 Text Analyzer

An advanced, asynchronous generative AI application that processes long documents, transcripts, and meeting notes to extract structured insights. 

Built with **LangGraph** and **FastAPI** on the backend, and **Gradio** on the frontend, this project demonstrates a highly efficient, decoupled architecture that executes large language model (LLM) tasks in parallel.

## ✨ Features

* **Parallel AI Execution:** Utilizes LangGraph to concurrently run 4 distinct LLM extraction tasks (Title, Summary, Action Items, Key Decisions), cutting processing time by up to 75% compared to sequential chains.
* **Structured Outputs:** Forces the LLM (Google Gemini) to return strictly typed JSON using Pydantic schemas, ensuring predictable and reliable data extraction.
* **Decoupled Architecture:** A standalone FastAPI backend (`/analyze` endpoint) that can be consumed by any client, paired with a lightweight Gradio frontend.
* **Asynchronous UI:** Features a live, non-blocking timer in the Gradio interface using Python background threads and websockets (`yield`).
* **PDF Export:** Automatically compiles the extracted markdown insights into a clean, downloadable PDF report on the fly.

## 🏗️ Architecture

1. **Frontend (`ui.py`):** Gradio web interface captures user text and sends a POST request to the API. Runs a background thread to maintain a live UI timer.
2. **Backend API (`api.py`):** FastAPI server receives the payload, validates it, and triggers the LangGraph orchestration engine.
3. **Orchestration (`graph.py`):** LangGraph initializes the `AgentState` and branches into 4 parallel nodes.
4. **LLM Engine:** Each node uses the Gemini API via LangChain to generate specific content, updating the final state graph before returning the payload to the frontend.

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* A Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/executive-text-analyzer.git](https://github.com/yourusername/executive-text-analyzer.git)
   cd executive-text-analyzer#

