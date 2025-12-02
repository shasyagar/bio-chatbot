import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. LOAD MODEL (CPU-only, Streamlit Cloud compatible)
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

    # IMPORTANT: no bitsandbytes, no quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    return tokenizer, model


# ============================================================
# 2. TEXT GENERATION
# ============================================================

def generate_reply(tokenizer, model, system_prompt, history, user_msg):
    history = history[-3:]  # keep context small (for speed)

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
    ).to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,        # long answers
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = text.split("Assistant:")[-1].strip()
    return reply


# ============================================================
# 3. STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Bioinformatics & Medical Assistant", layout="wide")
st.title("🧬 Bioinformatics & Medical Research Chatbot")

st.caption("For research & educational use only — not medical advice.")

tokenizer, model = load_model()


# ============================================================
# 4. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a highly knowledgeable assistant trained in:
- Bioinformatics (RNA-seq, ATAC-seq, WGS, scRNA-seq)
- Alignment, quantification, normalization, DE analysis
- Tools: STAR, HISAT2, Bowtie2, BWA, GATK, FastQC, Salmon, CellRanger
- QC, clustering, PCA, UMAP, ML methods
- Pathways, gene regulation, transcription factors
- Molecular biology, biochemistry, immunology
- Pharmacology and drug mechanisms (educational only)

Rules:
- DO NOT provide diagnosis or treatment.
- DO NOT make clinical decisions.
- Explain concepts scientifically and educationally.
"""


# ============================================================
# 5. CHAT INTERFACE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

# Display past messages
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask your bioinformatics or biomedical question…")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = generate_reply(
                tokenizer, model, SYSTEM_PROMPT,
                st.session_state.history[:-1], user_input
            )
            st.markdown(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
