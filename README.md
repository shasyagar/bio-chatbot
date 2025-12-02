# bio-chatbot
Bioinformatics &amp; Medical Knowledge Assistant

# 🧬 Bioinformatics & Medical Research Assistant  
*A lightweight, knowledge-rich AI assistant for bioinformatics and biomedical science.*

This project is a **Streamlit-based web application** powered by the **Phi-3 Mini LLM** (4bit quantized), capable of answering questions across:

- Bioinformatics  
- Genomics & transcriptomics  
- Next-generation sequencing (RNA-seq, ATAC-seq, WGS, scRNA-seq)  
- Computational biology  
- Molecules, proteins, and cellular pathways  
- Medical science & pharmacology (non-diagnostic, educational only)

The app is designed to be lightweight enough to run on **Streamlit Cloud (CPU only)** while providing accurate scientific knowledge and smooth chat interaction.

💡 **Important:**  
This assistant does **NOT** provide medical advice, diagnosis, or treatment.  
It is strictly for **research and educational use.**

---

## 🚀 Live Demo

Once deployed on Streamlit Cloud, your app will be live at a URL like:

https://your-app-name.streamlit.app


(Replace above after deployment.)

---

## 🛠️ Features

✔ Fast chat-style interface  
✔ Built-in scientific system prompt for bioinformatics & medical knowledge  
✔ Supports long, structured answers  
✔ Runs entirely on CPU using 4-bit quantized Phi-3 Mini  
✔ Suitable for research groups, students, and biomedical learners  
✔ 100% free to host on Streamlit Cloud  

---

## 📦 Tech Stack

- **Streamlit** (UI)
- **Transformers** (Hugging Face)
- **Phi-3 Mini 4K Instruct** (LLM)
- **BitsAndBytes** (4-bit quantization)
- **PyTorch** (CPU)

---

## 📁 Repository Structure

bioinformatics-chatbot/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
└── README.md # Documentation


---

## ⚙️ Installation (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/shasyagar/bio-chatbot.git
cd bio-chatbot

### 2. Create a virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the app
streamlit run app.py


