from collections import deque

SYSTEM_INSTRUCTION = (
    "System: You are a concise CLI assistant for geography Q&A. "
    "If the user's new message is under-specified (e.g., 'what about India?'), "
    "infer the missing attribute from the most recent user question and answer that analogous question. "
    "Do not include role labels like 'User:' or 'Assistant:' in your reply."
)

class ChatMemory:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.memory = deque(maxlen=window_size)

    def add_exchange(self, user_input, bot_response):
        self.memory.append({"user": user_input, "bot": bot_response})

    def last_user(self):
        for ex in reversed(self.memory):
            if ex.get("user"):
                return ex["user"]
        return ""

    def get_context(self):
        if not self.memory:
            return ""
        lines = []
        for ex in self.memory:
            lines.append(f"User: {ex['user']}")
            lines.append(f"Assistant: {ex['bot']}")
        return "\n".join(lines)

    def get_prompt_with_context(self, current_input):
        ctx = self.get_context()
        if ctx:
            return f"{SYSTEM_INSTRUCTION}\n\n{ctx}\nUser: {current_input}\nAssistant:"
        return f"{SYSTEM_INSTRUCTION}\n\nUser: {current_input}\nAssistant:"

    def clear_memory(self):
        self.memory.clear()

    def get_memory_size(self):
        return len(self.memory)
