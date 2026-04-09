# RAG Legal Assistant - Technical Synopsis

## 1. System Architecture

The application is a **Retrieval-Augmented Generation (RAG)** system built to serve as an expert Indian Legal Assistant. It follows a classic Client-Server architecture with a decoupled frontend and backend.

### High-Level Components
*   **Frontend**: Streamlit (`frontend/app.py`)
    *   Responsible for UI rendering, chat input, and session state management.
    *   Communicates with the backend via REST API.
*   **Backend**: FastAPI (`backend/main.py`)
    *   Exposes API endpoints.
    *   Manages the RAG pipeline and LLM orchestration.
*   **Data & Storage**:
    *   **Vector Database**: ChromaDB (Persistent storage on disk).
    *   **Raw Data**: PDF Documents in `data/`.
    *   **Embeddings**: `sentence-transformers/all-mpnet-base-v2` (Local inference).

---

## 2. End-to-End Data Flow

The system operates in two distinct phases: **Startup (Indexing)** and **Runtime (Chat)**.

### Phase A: Startup & Indexing Flow
When the backend starts (`backend/main.py` -> `@app.on_event("startup")`), it initializes the knowledge base.

```mermaid
graph TD
    A[Server Start] --> B{Check Data Directory}
    B -->|Found| C{Check Index Signature}
    B -->|Missing| Z[Error]
    
    C -->|Signature Matches| D[Load Existing ChromaDB]
    C -->|Signature Mismatch/Missing| E[Rebuild Index]
    
    E --> F[Scan PDFs]
    F --> G[Load & Parse PDFs]
    G --> H[Chunk Text (800 chars)]
    H --> I[Generate Embeddings (All-MPNet)]
    I --> J[Save to ChromaDB]
    J --> K[Update Signature JSON]
    
    D --> L[Load Documents for BM25]
    K --> L
    L --> M[Build In-Memory BM25 Index]
    M --> N[Initialize Ensemble Retriever]
    N --> O[Ready for Queries]
```

**Key Technical Details:**
1.  **Smart Indexing**: The system uses an `index_signature.json` file to store the state (modified time) of PDF files. It only rebuilds the expensive vector index if files have changed.
2.  **Hybrid Index Construction**:
    *   **Vector Index**: Built using ChromaDB and `all-mpnet-base-v2`.
    *   **Keyword Index**: Built in-memory using a custom `SimpleBM25Retriever`.
    *   *Note*: The BM25 index is rebuilt in-memory on every restart (fast operation), while the Vector index is persisted (slow operation).

### Phase B: Chat Request Lifecycle
When a user sends a message, the system decides how to handle it based on intent.

```mermaid
sequenceDiagram
    participant User
    participant Frontend (Streamlit)
    participant Backend (FastAPI)
    participant Router (Smart Routing)
    participant Retriever (Hybrid)
    participant LLM (Groq)

    User->>Frontend: Types "What is the verdict in Case 123?"
    Frontend->>Backend: POST /api/chat {session_id, message}
    Backend->>Router: _is_general_query(message)?
    
    alt General Greeting (e.g., "Hi")
        Router-->>Backend: Yes
        Backend->>Backend: Set Context = None
        Backend->>LLM: Send GENERAL_SYSTEM_PROMPT
    else Specific Question
        Router-->>Backend: No
        Backend->>Retriever: invoke(query)
        parallel
            Retriever->>Retriever: Run BM25 (Keyword Search)
            Retriever->>Retriever: Run Vector Search (Semantic)
        end
        Retriever->>Backend: Return Top Documents (RRF Fusion)
        Backend->>Backend: Construct RAG Prompt\n(System + Context + Query)
        Backend->>LLM: Send RAG_SYSTEM_PROMPT
    end
    
    LLM-->>Backend: Response (Content)
    Backend-->>Frontend: JSON {response: "..."}
    Frontend-->>User: Displays Message
```

---

## 3. Deep Dive: The RAG Engine

The core intelligence lies in `backend/services/rag.py`.

### 1. Hybrid Retrieval Strategy
The system acknowledges that legal search requires both specific precision (e.g., "Section 302") and semantic understanding (e.g., "murder definition").
*   **Vector Search (50% weight)**: Uses Cosine Similarity to find conceptually similar text. Good for "What are the arguments regarding X?".
*   **BM25 Search (50% weight)**: Uses probabilistic term matching. Essential for finding exact case numbers, dates, or specific legal codes that vector models might "blur".
*   **Ensemble**: The results are combined using **Reciprocal Rank Fusion (RRF)** to produce a final ranked list of context chunks.

### 2. Document Processing
*   **Loader**: `PyPDFLoader` handles raw PDF extraction.
*   **Splitting**: `RecursiveCharacterTextSplitter` handles text segmentation.
    *   **Chunk Size**: 800 characters.
    *   **Overlap**: 200 characters (ensures context continuity across boundaries).
*   **Metadata**: Original filenames and page numbers are preserved to enable citation in the final answer.

---

## 4. LLM & Inference Layer

Defined in `backend/api/chat.py`.

### 1. Model Provider
The project uses **Groq** for high-speed inference, communicating via the OpenAI-compatible API client.

### 2. Resilience Strategy
The system implements a robust fallback mechanism to handle API outages or rate limits:
1.  **Primary**: `llama-3.3-70b-versatile` (High intelligence, good for legal reasoning).
2.  **Fallback 1**: `openai/gpt-oss-120b` (Alternative high-param model).
3.  **Fallback 2**: `meta-llama/llama-4-maverick...` (Smaller, faster model).

### 3. Prompt Engineering
*   **System Prompt**: Enforces a strict "Source of Truth" policy. The model is explicitly forbidden from using outside knowledge and must state if the document does not contain the answer. This helps prevent legal hallucinations.

## 5. Session Management
*   **State Location**: In-memory Python dictionary (`_chat_sessions`).
*   **Concurrency**: Protected by `threading.Lock()` to ensure thread safety during concurrent requests.
*   **History**: Retains the last 40 turns of conversation to maintain context without overflowing the context window.
