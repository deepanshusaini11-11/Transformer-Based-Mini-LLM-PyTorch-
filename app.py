import streamlit as st
import torch
import os
from mini_llm import MiniLLM, build_tokenizer, BLOCK_SIZE, DEVICE

st.set_page_config(page_title="Mini LLM", page_icon="🤖")
st.title("Mini LLM — Shakespeare Generator")

@st.cache_resource
def load_model():
    text = open("input.txt").read()
    encode, decode, vocab_size = build_tokenizer(text)
    model = MiniLLM(vocab_size).to(DEVICE)
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
        model.eval()
    return model, encode, decode

model, encode, decode = load_model()

prompt      = st.text_area("Enter a prompt", value="To be or not to be")
max_tokens  = st.slider("Tokens to generate", 50, 500, 200)
temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1)

if st.button("Generate"):
    with st.spinner("Generating..."):
        try:
            context = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                out = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
            st.text_area("Output", decode(out[0].tolist()), height=300)
        except KeyError as e:
            st.error(f"Prompt contains a character not seen in training: {e}")
