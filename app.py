import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ===============================
# Load model
# ===============================
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-3-mini-4k-instruct"

    quant = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        quantization_config=quant,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    return tokenizer, model

# ===============================
# Generation
# ===============================
def generate_reply(tokenizer, model, system_prompt, history, user_input):

    history = history[-3:]

    convo = [f"System: {system_prompt}"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        convo.append(f"{role}: {msg['content']}")
    convo.append(f"User: {user_input}")
    convo.append("Assistant:")

    prompt = "\n".join(convo)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("Assistant:")[-1].strip()


# ===============================
# Streamlit App UI
# ===============================
st.set_page_config(page_title="Bioinformatics & Medical Assistant", layout="wide")
st.title("🧬 Bioinformatics & Medical Research Assistant")

tokenizer, model = load_model()

SYSTEM_PROMPT = """
You are a highly knowledgeable assistant trained in:
- Bioinformatics (RNA-seq, ATAC-seq, WGS, scRNA-seq)
- Tools: STAR, HISAT2, Bowtie2, BWA, GATK, FastQC, Salmon, CellRanger
- Differential expression, clustering, PCA, QC, ML for omics
- Pathways, gene regulation, molecular biology
- Physiology, immunology, pharmacology (knowledge only)

Rules:
- DO NOT provide diagnosis or treatment.
- DO NOT make clinical decisions.
- Provide science explanations only.
"""

if "history" not in st.session_state:
    st.session_state.history = []

# Show history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything about bioinformatics or medical science...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_reply(tokenizer, model, SYSTEM_PROMPT, st.session_state.history[:-1], user_input)
            st.markdown(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
