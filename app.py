# app.py
import json
import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
import PyPDF2
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ===========================
# Load Qwen Model
# ===========================
model_name = "mistralai/qwen-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# ===========================
# Load PDF Data
# ===========================
pdf_text = ""
try:
    with open("kampus.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pdf_text += page.extract_text()
except FileNotFoundError:
    pdf_text = ""
    print("kampus.pdf tidak ditemukan, lanjut tanpa PDF.")

# ===========================
# Load JSON Data
# ===========================
json_data = {}
try:
    with open("data.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
except FileNotFoundError:
    json_data = {}
    print("data.json tidak ditemukan, lanjut tanpa JSON.")

# ===========================
# Streamlit UI
# ===========================
st.title("Chatbot Kampus - Qwen AI")

prompt = st.text_input("Masukkan pertanyaan:")

def generate_answer(user_prompt):
    # Gabungkan semua knowledge base
    full_prompt = f"{user_prompt}\n\nReferensi:\n"
    if pdf_text:
        full_prompt += pdf_text + "\n"
    if json_data:
        full_prompt += json.dumps(json_data, ensure_ascii=False) + "\n"

    # Generate jawaban pakai Qwen
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if st.button("Kirim"):
    with st.spinner("AI sedang memproses..."):
        answer = generate_answer(prompt)
        st.text_area("Jawaban AI:", value=answer, height=200)

# ===========================
# Flask API
# ===========================
api_app = Flask(__name__)

@api_app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_prompt = data.get("prompt", "")
    answer = generate_answer(user_prompt)
    return jsonify({"response": answer})

def run_flask():
    api_app.run(port=5001)

# Jalankan Flask API di background
Thread(target=run_flask).start()
