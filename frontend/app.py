import uuid
import streamlit as st
import requests

st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("RAG legalDesk AI")

import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/chat/chat")

print(BACKEND_URL)
# -------------------------------------------------------------------
# Session-scoped state
# -------------------------------------------------------------------
# Unique session id per browser session (used by backend to track history)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Chat messages just for UI rendering on the frontend
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------------------------
# Chat history display
# -------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------------------------------------------
# Chat input
# -------------------------------------------------------------------
prompt = st.chat_input("Ask something about your document...")

if prompt:
    # 1) Show user message in UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Call backend with session_id + latest message
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                }
                response = requests.post(BACKEND_URL, json=payload)
                print(response)

                if response.status_code == 200:
                    ai_reply = response.json().get("response", "")
                    if not ai_reply:
                        ai_reply = "_Empty response from backend_"

                    st.markdown(ai_reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": ai_reply}
                    )
                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
