# RAG Legal Assistant - Detailed Project Report

## 1. Executive Summary
The **RAG Legal Assistant** is a specialized AI application designed to help legal professionals and users navigate complex Indian legal documents. It leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, citation-backed answers from a library of PDF documents (such as court judgments), ensuring that the AI's responses are grounded strictly in the provided evidence rather than general knowledge.

## 2. System Architecture
The system follows a modern **Client-Server** architecture:

*   **Frontend (User Interface)**: Built with **Streamlit**, providing a clean, browser-based chat interface where users can type queries and view conversation history.
*   **Backend (API Layer)**: Built with **FastAPI**, serving as the brain of the operation. It handles request routing, session management, and the core RAG logic.
*   **Database (Knowledge Base)**: Uses **ChromaDB**, a vector database, to store and potential meanings of document text (chunks).
*   **AI Engine (Inference)**: Powered by **Groq** (using LLaMA 3.3 models) for ultra-fast text generation.

## 3. Detailed Data Flow

### Phase 1: Data Ingestion (The "Learning" Phase)
Before the system can answer questions, it must "read" the documents. This happens automatically when the server starts:
1.  **Scanning**: The system checks the `data/` folder for PDF files.
2.  **Smart Caching**: It compares the current files against a saved `index_signature.json`. If files haven't changed, it skips processing to save time.
3.  **Processing**:
    *   **Loading**: Using `PyPDFLoader`, it extracts text from every page.
    *   **Chunking**: The text is split into smaller "chunks" of 800 characters. This ensures the AI gets precise context, not just whole pages.
4.  **Indexing (Hybrid Approach)**:
    *   **Vector Index**: Chunks are converted into mathematical vectors (embeddings) using `sentence-transformers/all-mpnet-base-v2` and saved in ChromaDB. This allows the system to understand *concepts* (e.g., "murder" is related to "homicide").
    *   **Keyword Index**: A BM25 index is built in-memory. This allows the system to find *exact matches* (e.g., "Case No. 12345/2023").

### Phase 2: Query Resolution (The "Chat" Phase)
When a user asks a question, the system follows this logic:
1.  **Intent Detection**: The system analyzes the message.
    *   If it's a greeting (e.g., "Hi", "Good morning"), it responds instantly with a greeting.
    *   If it's a question, it activates the RAG pipeline.
2.  **Hybrid Retrieval**:
    *   The system searches the Vector Index for conceptually similar text.
    *   Simultaneously, it searches the Keyword Index for exact term matches.
    *   It combines these results (Ensemble Search) to get the "best of both worlds"—conceptual understanding and factual precision.
3.  **Prompt Assembly**:
    *   The system collects the top-ranked text chunks.
    *   It creates a strict instruction for the AI: *"Answer the user's question using ONLY these text chunks. Do not use outside knowledge."*
4.  **Generative Response**:
    *   The prompt is sent to the **Groq AI API**.
    *   If the primary model fails, the system automatically tries backup models.
    *   The generated answer is sent back to the user.

## 4. Key Technical Features

### Smart Persistence
The system is respectful of resources. It writes the heavy vector database to disk (`persist_directory`) so that it doesn't need to re-read and re-calculate thousands of document pages every time you restart the server. It only rebuilds when it detects new or modified files.

### Hallucination Control
Legal AI must be accurate. The system uses a strict "System Prompt" that forces the AI to admit ignorance if the answer isn't in the chunks. This prevents the AI from inventing plausible-sounding but false legal facts.

### Efficient Session Management
Chat history is stored in-memory on the server, linked to a specific session ID. The system keeps a sliding window of the last 40 messages, ensuring the conversation flows naturally without exceeding the AI's memory limits.

## 5. Technology Stack Summary
*   **Language**: Python 3.10+
*   **Web Frameworks**: FastAPI (Backend), Streamlit (Frontend)
*   **AI/ML Libraries**: LangChain, ChromaDB, Sentence-Transformers
*   **External APIs**: Groq (LLaMA models)
