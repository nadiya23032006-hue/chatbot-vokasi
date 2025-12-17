# app.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread

# ===========================
# Load Qwen model
# ===========================
model_name = "mistralai/qwen-7b"  # ganti sesuai model yang kamu mau
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# ===========================
# Streamlit UI
# ===========================
st.title("Qwen Chat AI - Streamlit")

prompt = st.text_input("Masukkan pertanyaan:")

if st.button("Kirim"):
    with st.spinner("AI sedang memproses..."):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=150)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("Jawaban AI:", value=result, height=200)

# ===========================
# Flask API
# ===========================
api_app = Flask(__name__)

@api_app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": result})

def run_flask():
    api_app.run(port=5001)

# Jalankan Flask API di background
Thread(target=run_flask).start()
