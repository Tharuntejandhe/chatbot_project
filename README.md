# Local Command-Line Chatbot with Sliding Window Memory

A fully functional local chatbot interface using Hugging Face text generation models with sliding window memory for coherent multi-turn conversations. Runs entirely on CPU without GPU requirements.

## Features

- **CPU-Optimized**: Runs on any laptop without GPU
- **Conversational Memory**: Maintains context using sliding window (5 turns)
- **Intent Resolution**: Understands elliptical follow-ups like "what about X?"
- **Modular Architecture**: Clean separation of concerns across 3 modules
- **Easy Configuration**: Switch models and adjust parameters easily

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python interface.py
```

## Usage Examples

```
User: What is the capital of France?
Assistant: The capital of France is Paris.

User: What about Italy?
Assistant: The capital of Italy is Rome.

User: /exit
Exiting chatbot. Goodbye!
```

## Project Structure

```
chatbot_project/
model_loader.py       # Model and tokenizer loading.    chat_memory.py        # Sliding window memory buffer
interface.py          # CLI loop and integration
requirements.txt      # Dependencies
README.md            # This file
```

## Configuration

### Change Model
Edit `interface.py` line 95:
```python
chatbot = ChatbotInterface(
    model_name="HuggingFaceTB/SmolLM-360M-Instruct",  # Change here
    memory_window=5
)
```

### Available Models

| Model | Parameters | RAM | Speed |
|-------|-----------|-----|-------|
| TinyLlama-1.1B (default) | 1.1B | 2-3 GB | Fast |
| SmolLM-360M | 360M | 1-2 GB | Very Fast |
| distilgpt2 | 82M | <1 GB | Extremely Fast |

## Troubleshooting

**Out of Memory**: Use smaller model (`distilgpt2`)  
**Slow Responses**: Reduce `max_new_tokens` to 40  
**Indentation Errors**: Ensure no leading spaces before imports

## License

MIT License - see LICENSE file for details.
