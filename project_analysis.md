# RAG Legal Assistant - Project Analysis

## Overview
The **RAG Legal Assistant** is a specialized AI application designed to provide citation-backed answers from legal documents using Retrieval-Augmented Generation (RAG). 

## Architecture
The project follows a standard **Client-Server** architecture with a modern stack:

### Backend
- **Framework**: FastAPI (`backend/main.py`)
- **API & Routing**:
  - `backend/api/endpoints.py`: General API endpoints.
  - `backend/api/chat.py`: Handles chat completions, session history logic, and smart routing (Greeting vs RAG).
- **Services**:
  - `backend/services/rag.py`: Core RAG logic, implementing **Hybrid Search** (Vector + BM25) and index management.

### Frontend
- **Current Approach**: Custom HTML/JS application served statically.
  - `backend/templates/index.html`: Main chat interface.
  - `backend/static/script.js`: Client-side logic for chat interaction and API calls.
  - `backend/static/style.css`: Styling for the chat interface.
- **Legacy/Prototype**: Streamlit app (`frontend/app.py`), likely being replaced or kept for administration.

### Data & Persistence
- **Vector Database**: ChromaDB (persisted in `data/chroma_db`).
- **Documents**: PDF files stored in `data/`.
- **Indexing**: A smart signature system (`index_signature.json`) prevents unnecessary re-indexing by checking file timestamps and config.

## Key Features
1. **Hybrid Search**: Combines **Vector Search** (semantic) and **BM25** (keyword) using Reciprocal Rank Fusion (RRF) for high-precision retrieval.
2. **Smart Caching**: `_build_retriever` checks the `index_signature.json` to decide whether to rebuild the vector index, saving startup time.
3. **Session Management**: In-memory chat history (`_chat_sessions` in `chat.py`) with a sliding window to manage context length.
4. **Resiliency**: The chat endpoint includes a retry mechanism for model calls (`MODELS_TO_TRY`) to handle API failures gracefully.

## Current State & Recommendations
- **Frontend Integration**: The user is currently verifying the integration of the HTML/JS frontend (`script.js` is active). The `index.html` correctly links resources.
- **Robustness**: The backend logic for RAG is sophisticated, handling edge cases like "no docs found" or "all models failed" well.
- **Scaling**: Currently, session memory is local to the server process. For a production deployment with multiple workers, a Redis-backed session store would be needed.

## Next Steps
- Verify the `script.js` logic handles all UI states (loading, error, markdown rendering).
- Ensure `style.css` provides the desired dark/light theme experience.
- Test the full end-to-end flow: `User -> Frontend -> FastAPI -> RAG Service -> Groq -> Frontend`.
