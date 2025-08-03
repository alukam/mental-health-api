from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from wordsegment import load, segment

app = Flask(__name__)

# Load model and tokenizer
model_path = "t5_mental_health_model"  # change if different
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


load()  # Only needs to be called once

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.get_json()
    user_input = data["text"]

    input_ids = tokenizer("question: " + user_input, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=128,
        no_repeat_ngram_size=3,  # ✨ prevents repeating 3-word phrases
        num_beams=5,              # ✨ improves quality
        early_stopping=True
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    spaced_output = " ".join(segment(output_text))

    return jsonify({"response": spaced_output})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

    