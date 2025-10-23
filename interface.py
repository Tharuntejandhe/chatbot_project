from model_loader import ModelLoader
from chat_memory import ChatMemory

STOP_MARKERS = ["\nUser:", " User:", "\nAssistant:", "\nBot:", "\nSystem:", "\nHuman:", "\nQ:"]

def _clean_reply(text: str) -> str:
    prefixes = ["Assistant:", "Bot:", "AI:", "System:"]
    t = text.strip()
    for p in prefixes:
        if t.startswith(p):
            t = t[len(p):].strip()
            break
    cuts = [t.find(s) for s in STOP_MARKERS if s in t]
    if cuts:
        t = t[:min(cuts)].strip()
    return " ".join(t.split())

# Simple intent resolver for elliptical follow-ups like "what about India?"
def resolve_intent(last_user_q: str, current_user_q: str) -> str:
    if not last_user_q:
        return current_user_q.strip()
    cu = current_user_q.strip().lower()
    # Only attempt rewrite if the new question is under-specified
    if cu in {"what about india?", "what about india", "what about", "what about it?", "what about it"} or cu.startswith("what about "):
        lu = last_user_q.lower()
        # Map common attributes
        if "capital" in lu or "capital city" in lu:
            # Extract entity after "what about"
            after = current_user_q.strip()[len("what about"):].strip(" ?.")
            return f"What is the capital city of {after}?"
        if "largest city" in lu:
            after = current_user_q.strip()[len("what about"):].strip(" ?.")
            return f"What is the largest city in {after}?"
        if "population" in lu:
            after = current_user_q.strip()[len("what about"):].strip(" ?.")
            return f"What is the population of {after}?"
        if "area" in lu:
            after = current_user_q.strip()[len("what about"):].strip(" ?.")
            return f"What is the area of {after}?"
    return current_user_q.strip()

class ChatbotInterface:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", memory_window=5):
        self.model_loader = ModelLoader(model_name)
        self.memory = ChatMemory(window_size=memory_window)
        self.running = False

    def initialize(self):
        print("=" * 60)
        print("Initializing Chatbot...")
        print("=" * 60)
        self.model_loader.load_model()
        print("=" * 60)
        print("Chatbot ready! Type '/exit' to quit.")
        print("=" * 60)

    def generate_response(self, user_input):
        # Rewrite under-specified follow-ups using last user question
        last_u = self.memory.last_user()
        effective_input = resolve_intent(last_u, user_input)
        prompt = self.memory.get_prompt_with_context(effective_input)
        raw = self.model_loader.generate_response(
            prompt,
            max_new_tokens=60,
            temperature=0.3,
            top_k=40,
            top_p=0.9
        )
        return _clean_reply(raw)

    def run(self):
        self.running = True
        while self.running:
            try:
                user_input = input("\nUser: ").strip()
                if user_input.lower() == "/exit":
                    print("\nExiting chatbot. Goodbye!")
                    self.running = False
                    break
                if not user_input:
                    continue
                print("Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                self.memory.add_exchange(user_input, response)
            except KeyboardInterrupt:
                print("\n\nExiting chatbot. Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

    def display_info(self):
        info = self.model_loader.get_model_info()
        print("\nChatbot Information:")
        print(f"Model: {info['model_name']}")
        print(f"Device: {info['device']}")
        print(f"Memory Window: {self.memory.window_size} turns")
        print(f"Current Memory: {self.memory.get_memory_size()} turns")

def main():
    chatbot = ChatbotInterface(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", memory_window=5)
    chatbot.initialize()
    chatbot.display_info()
    chatbot.run()

if __name__ == "__main__":
    main()
