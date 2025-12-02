import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Load Model (Streamlit Cloud compatible)
# ============================================================
@st.cache_resource
def load_model():
    model_name = "microsoft/phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # IMPORTANT: No device_map, no low_cpu_mem_usage, no quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    return tokenizer, model


# ============================================================
# Generate reply
# ============================================================
def generate_reply(tokenizer, model, system_prompt, history, user_msg):
    history = history[-3:]

    convo = [f"System: {system_prompt}"]
    for h in history:
        role = "User" if h["role"] == "user" else "Assistant"
        convo.append(f"{role}: {h['content']}")
    convo.append(f"User: {user_msg}")
    convo.append("Assistant:")

    prompt = "\n".join(convo)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Assistant:")[-1].strip()


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Bioinformatics & Medical Assistant", layout="wide")
st.title("🧬 Bioinformatics & Medical Science Assistant")
st.caption("For research & educational use only — not medical advice.")

tokenizer, model = load_model()

SYSTEM_PROMPT = """
You are a highly knowledgeable assistant trained in:
- Bioinformatics (RNA-seq, ATAC-seq, WGS, scRNA-seq)
- Tools: STAR, HISAT2, Bowtie2, BWA, GATK, FastQC, Salmon, CellRanger
- Differential expression, normalization, QC, clustering
- PCA, UMAP, ML methods
- Pathways, gene regulation, biology, immunology
- Pharmacology and drug mechanisms (educational only)

Rules:
- Do NOT diagnose diseases.
- Do NOT recommend treatments.
- Provide scientific, educational explanations only.
"""

if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_msg = st.chat_input("Ask your bioinformatics or biomedical science question…")

if user_msg:
    st.session_state.history.append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate_reply(
                tokenizer, model, SYSTEM_PROMPT,
                st.session_state.history[:-1], user_msg
            )
            st.markdown(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
