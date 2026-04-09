document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Generate or retrieve session ID
    let sessionId = localStorage.getItem('rag_session_id');
    if (!sessionId) {
        sessionId = crypto.randomUUID();
        localStorage.setItem('rag_session_id', sessionId);
    }

    console.log("Session ID:", sessionId);

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value === '') {
            this.style.height = 'auto';
        }
    });

    // Send on Enter (but Shift+Enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Add user message
        appendMessage('user', text);
        userInput.value = '';
        userInput.style.height = 'auto';

        // Add loading state
        const loadingId = appendLoadingMessage();

        try {
            const response = await fetch('/api/chat/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: text
                })
            });

            // Remove loading message
            removeMessage(loadingId);

            if (response.ok) {
                const data = await response.json();
                const aiReply = data.response;
                appendMessage('assistant', aiReply);
            } else {
                const errorData = await response.json();
                appendMessage('assistant', `**Error:** ${errorData.detail || 'Something went wrong.'}`);
            }

        } catch (error) {
            removeMessage(loadingId);
            appendMessage('assistant', `**Connection Error:** ${error.message}`);
        }
    }

    function appendMessage(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        // Parse markdown using marked.js
        bubble.innerHTML = marked.parse(content);

        msgDiv.appendChild(bubble);
        chatContainer.appendChild(msgDiv);

        scrollToBottom();
        return msgDiv;
    }

    function appendLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.id = id;

        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.innerHTML = '<em>Thinking...</em>';

        msgDiv.appendChild(bubble);
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
