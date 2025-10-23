from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class ModelLoader:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.generator = None

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self):
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device if self.device != "cpu" else -1
        )
        print("Model loaded successfully!")

    def generate_response(
        self,
        prompt,
        max_new_tokens=60,
        temperature=0.3,
        top_k=40,
        top_p=0.9
    ):
        if self.generator is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )
        text = outputs[0]["generated_text"].strip()
        return text

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.model is not None
        }
