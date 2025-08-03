from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model and tokenizer once
MODEL_PATH = "model/t5_mental_health_model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.eval()

def generate_response(user_input):
    input_text = f"question: {user_input}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=256, do_sample=True, top_k=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
