from flask import Flask, request, jsonify
import torch
from model import MiniTransformer

app = Flask(__name__)

# Load text
with open("sample.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

vocab_size = len(chars)

# Load model
model = MiniTransformer(vocab_size)
model.load_state_dict(torch.load("mini_llm.pth"))

model.eval()

@app.route("/")
def home():
    return "Mini Transformer LLM Running"

@app.route("/generate", methods=["POST"])
def generate():

    data = request.json
    start = data["text"]

    input_seq = torch.tensor(
        [stoi[c] for c in start],
        dtype=torch.long
    ).unsqueeze(1)

    for _ in range(100):

        output = model(input_seq)

        predicted = torch.argmax(
            output[-1],
            dim=-1
        ).item()

        input_seq = torch.cat(
            [input_seq, torch.tensor([[predicted]])],
            dim=0
        )

    result = ''.join(
        [itos[i.item()] for i in input_seq.squeeze()]
    )

    return jsonify({"generated_text": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
