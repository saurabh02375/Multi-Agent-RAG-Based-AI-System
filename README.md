# Multi-Agent-RAG-Based-AI-System

## Project Overview

**Multi-Agent RAG-Based AI System** is a retrieval-augmented generation solution focused on legal document intelligence. It blends vector search, keyword matching, and multi-agent orchestration to deliver precise, citation-aware answers from legal PDFs and structured data.

### What this project does
- Ingests legal documents from the `data/` folder.
- Builds and maintains a semantic vector index using ChromaDB.
- Executes hybrid retrieval through both semantic search and BM25 ranking.
- Generates contextual answers using LLM-based completion logic.
- Serves a responsive web UI through a FastAPI backend.

## Key Features

- **Hybrid Retrieval Strategy**
  - Vector search for semantic relevance.
  - BM25 keyword search for exact legal term matching.
  - Reciprocal Rank Fusion (RRF) to combine result rankings and increase precision.

- **Retrieval-Augmented Generation (RAG)**
  - Pulls document snippets from retrieved sources.
  - Builds a context-aware prompt for the language model.
  - Returns citation-backed responses that reference the source PDFs.

- **Session-aware Chat Interface**
  - Stores short-term chat history in memory.
  - Supports multi-turn conversations with context preservation.
  - Differentiates between greetings and legal assistant queries.

- **Smart Index Management**
  - Uses `data/index_signature.json` to detect changes in source documents.
  - Rebuilds the index only when the data set or configuration changes.

## Architecture

### Backend
- **Framework**: FastAPI
- **Main app**: `backend/main.py`
- **API modules**:
  - `backend/api/chat.py` — chat handling, session logic, and RAG request flow.
  - `backend/api/endpoints.py` — general API routes and health checks.
  - `backend/api/schemas.py` — request/response validation models.
- **RAG service**:
  - `backend/services/rag.py` — index creation, retrieval, and prompt assembly.

### Frontend
- **Static web UI** served by FastAPI
- **Templates**: `backend/templates/index.html`
- **Static assets**:
  - `backend/static/script.js`
  - `backend/static/style.css`
- **Prototype app**: `frontend/app.py` (Streamlit-based, optional exploration mode)

### Data
- Document storage: `data/`
- Vector DB persistence: `data/chroma_db/`
- Index signature: `data/index_signature.json`

## Folder Structure

```text
backend/
  main.py
  api/
    chat.py
    endpoints.py
    schemas.py
  services/
    rag.py
  static/
    script.js
    style.css
  templates/
    index.html
frontend/
  app.py
data/
  chroma_db/
  index_signature.json
  *.pdf
```

## Getting Started

1. Create a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Set any required environment variables for API keys.
4. Run the backend with Uvicorn or `python backend/main.py`.

## Recommended Commands

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python backend/main.py
```

## Usage

- Open the web UI in a browser.
- Ask questions related to the loaded legal documents.
- The system will return answers with sources and citations.

## Future Improvements

- Add Redis-backed session persistence for multi-worker deployments.
- Improve the frontend with a modern React or Vue interface.
- Add user authentication and audit logging for legal query histories.
- Enhance document ingestion with automatic PDF parsing and metadata extraction.

## Notes

- This repository is primarily a legal RAG prototype.
- The system is designed to be extended with multiple agents and smarter orchestration.
- Use the included `.gitignore` to keep secrets and local artifacts out of source control.
