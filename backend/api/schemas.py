from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request body for a chat completion.

    `session_id` groups messages into a logical conversation.
    `message` is the latest user input.
    """
    session_id: str = Field(..., description="Unique identifier for the chat session")
    message: str = Field(..., description="User's latest message")
