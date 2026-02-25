import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.config import MODEL_NAME, TEMPERATURE, TOP_K
from core.sampling import apply_temperature, top_k_filter

class LanguageModelEngine:

    def __init__(self):
        print("Loading AI language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model.eval()
        print("Model loaded.")

    def predict(self, text):

        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

        logits = apply_temperature(logits, TEMPERATURE)
        logits = top_k_filter(logits, TOP_K)

        probs = torch.softmax(logits, dim=-1)
        top_words = torch.topk(probs, 3)

        predictions = []

        for idx in top_words.indices[0]:
            predictions.append(
                self.tokenizer.decode([idx]).strip()
            )

        return predictions