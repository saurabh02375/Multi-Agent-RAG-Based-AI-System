"""
Chat endpoints for the RAG POC.

This module:
- Manages per-session in-memory chat history
- Calls Groq (via OpenAI client) for chat completions
- Keeps history trimmed to avoid token limit issues
- Implements Smart Routing (Greeting vs RAG) and Model Fallback
"""

import os
import time
from threading import Lock
import re
from typing import Dict, List, TypedDict, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import OpenAI, APIStatusError, RateLimitError
from backend.services.rag import get_retriever
from backend.api.schemas import ChatRequest

load_dotenv()

router = APIRouter()

# ---------------------------------------------------------------------------
# OpenAI / Groq client setup
# ---------------------------------------------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    # Fail fast instead of silently 500-ing on every request
    raise RuntimeError("OPENAI_API_KEY is not set. Check your .env configuration.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY,
)

# ---------------------------------------------------------------------------
# Models & Configuration
# ---------------------------------------------------------------------------

# Primary model (User requested: llama-3.3-70b-versatile)
PRIMARY_MODEL = "llama-3.3-70b-versatile"

# Fallback models as requested by user
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",  # Placeholder/Specific ID requested by user
    "meta-llama/llama-4-maverick-17b-128e-instruct" # Specific ID requested by user
]

MODELS_TO_TRY = [PRIMARY_MODEL] + FALLBACK_MODELS

# Detection for general queries (to skip RAG)
GREETING_KEYWORDS = {
    "hi", "hello", "hey", "hallo", "greetings", "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "bye", "goodbye"
}

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, friendly, and expert Legal Assistant.\n"
    "Your goal is to converse naturally with the user.\n"
    "If the user greets you, greet them back warmly and ask how you can help with their legal documents.\n"
    "Do NOT make up facts. If asked about specific documents, ask the user to provide details if they are not loaded."
)

RAG_SYSTEM_PROMPT = (
    "You are an expert Indian Legal Assistant. Your role is to analyze court judgments and legal documents with precision.\n\n"
    "CRITICAL INSTRUCTIONS FOR ACCURACY:\n"
    "1. **SOURCE OF TRUTH**: You must answer ONLY using the information present in the DOCUMENT CONTEXT below. Do not use outside knowledge.\n"
    "2. **NO HALLUCINATIONS**: If the answer is not explicitly in the text, say 'The provided documents do not contain this information.' do NOT make up an answer.\n"
    "3. **DIRECT & PROFESSIONAL**: Be concise. Remove filler phrases like 'Based on the provided text'. Just state the answer.\n"
    "4. **CITATIONS**: When referencing facts, implicitly rely on the source provided. If a specific case number or section is mentioned in the text, include it.\n\n"
    "DOCUMENT CONTEXT:\n"
    "─────────────────────────────────────\n"
    "{context}\n"
    "─────────────────────────────────────\n\n"
    "Question: {user_message}\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# Types for in-memory session store
# ---------------------------------------------------------------------------

class ChatMessage(TypedDict):
    role: str   # "user" | "assistant" | "system"
    content: str


# session_id -> [messages...]
_chat_sessions: Dict[str, List[ChatMessage]] = {}
_sessions_lock = Lock()

# To avoid hitting token limits, only send this many recent turns to the model.
HISTORY_LIMIT = 40

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session_history(session_id: str) -> List[ChatMessage]:
    """Return the full history for a session (may be empty)."""
    with _sessions_lock:
        return list(_chat_sessions.get(session_id, []))  # copy to avoid mutation bugs


def _append_to_history(session_id: str, message: ChatMessage) -> None:
    """Append a single message to a session's history."""
    with _sessions_lock:
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = []
        _chat_sessions[session_id].append(message)


def _is_general_query(query: str) -> bool:
    """Check if the query is a simple greeting or general conversation."""
    cleaned = query.strip().lower()
    # Check if exact match or starts with greeting (e.g. "hi there")
    if cleaned in GREETING_KEYWORDS:
        return True
    
    # Check for short generic phrases (simple heuristic)
    if len(cleaned.split()) < 3 and any(w in cleaned for w in GREETING_KEYWORDS):
        return True
        
    return False


def _build_model_messages(session_id: str, user_message: str, context: Optional[str] = None) -> List[ChatMessage]:
    """
    Build the list of messages to send to the model.
    Decides between General or RAG prompt based on presence of context.
    """
    history = _get_session_history(session_id)
    recent_history = history[-HISTORY_LIMIT:]

    if context:
        # RAG Mode
        system_content = RAG_SYSTEM_PROMPT.format(context=context, user_message=user_message)
    else:
        # General Mode
        system_content = GENERAL_SYSTEM_PROMPT

    system_msg: ChatMessage = {
        "role": "system",
        "content": system_content
    }

    messages: List[ChatMessage] = [
        system_msg,
        *recent_history,
        {"role": "user", "content": user_message},
    ]
    return messages


def _get_context_for_question(question: str, max_chars: int = 15000) -> Optional[str]:
    """
    Use the retriever to get relevant chunks for the given question.
    Returns None if no relevant context found or if retriever is unavailable.
    """
    # Remove inefficient raw fallback
    # We rely entirely on the vector store which supports multiple PDFs.

    # --- Vector store retrieval ---
    # --- Vector store retrieval ---
    retriever = get_retriever()
    if retriever is None:
        print(" [CHAT CHECK] Retriever is None!")
        return None

    try:
        print(f" [CHAT CHECK] Retrieving docs for: '{question}'")
        docs = retriever.invoke(question)
        print(f" [CHAT CHECK] Retrieved {len(docs)} docs.")
        
        if not docs:
            print(" [CHAT CHECK] No docs found.")
            return None
        
        # Concatenate without chunk numbers
        parts = []
        current_len = 0
        
        for i, doc in enumerate(docs):
            text = doc.page_content.strip()
            # Extract metadata (default to unknown if missing)
            source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
            page = doc.metadata.get("page", "??")
            
            # Format: [Source: file.pdf, Page: 5] Content...
            cited_text = f"[Source: {source}, Page: {page}]\n{text}"
            
            print(f" [CHAT CHECK] Doc {i} preview: {text[:50]}...") 
                
            if current_len + len(cited_text) > max_chars:
                print(f" [CHAT CHECK] Max chars reached ({max_chars}). Stopping.")
                break
                
            parts.append(cited_text)
            current_len += len(cited_text)
        
        if not parts:
            return None
            
        final_context = "\n\n".join(parts)
        print(f" [CHAT CHECK] Final context length: {len(final_context)}")
        return final_context
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return None


# ---------------------------------------------------------------------------
# Public endpoint
# ---------------------------------------------------------------------------

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint with Smart Routing and Model Fallback.
    """
    session_id = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        # 1. Determine Mode (General vs RAG)
        context = None
        if not _is_general_query(user_message):
            # Only fetch context if it's NOT a general greeting
            context = _get_context_for_question(user_message)

        # 2. Build Messages
        messages = _build_model_messages(session_id, user_message, context)

        # 3. Call Model with Fallback Logic
        ai_response = None
        last_error = None
        
        print(f"Attempting chat with {len(MODELS_TO_TRY)} models...")

        for model_name in MODELS_TO_TRY:
            try:
                print(f"Trying model: {model_name}")
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=0.0,
                    timeout=30.0, # Fail fast if model hangs
                )
                ai_response = chat_completion.choices[0].message.content
                if ai_response:
                    break # Success!
                    
            except (APIStatusError, RateLimitError) as e:
                # These are model/provider errors, worth trying next model
                print(f"Model {model_name} failed: {e}")
                last_error = e
                continue
            except Exception as e:
                # Unexpected system error, probably shouldn't just retry purely on this, 
                # but if it's network related, fallback might help? 
                # For safety, let's treat it as a failure of this attempt.
                print(f"Model {model_name} error: {e}")
                last_error = e
                continue

        if ai_response is None:
            # All models failed
            detail_msg = f"All models failed. Last error: {str(last_error)}"
            raise HTTPException(status_code=502, detail=detail_msg)

        # 4. Success - Save to History
        _append_to_history(session_id, {"role": "user", "content": user_message})
        _append_to_history(session_id, {"role": "assistant", "content": ai_response})

        return {"response": ai_response}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"CRITICAL CHAT ERROR: {e}")
        # Return a friendly fallback instead of 500
        return {"response": "I apologize, but I'm having trouble connecting to my brain right now. Please try again in a moment."}
