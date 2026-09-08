# 📚 HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

Users can upload an HR Policy PDF and ask questions about its contents.

## 🚀 Features

- Upload HR Policy PDF
- Extract text using PyMuPDF
- Split documents into chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers using Groq GPT-OSS 20B
- Display source pages
- Chat-style interface using Streamlit

## 🧠 RAG Architecture

PDF
↓
PyMuPDF
↓
Text Chunks
↓
Sentence Transformers
↓
FAISS
↓
Relevant Chunks
↓
Groq GPT-OSS 20B
↓
Answer + Sources

## 🛠️ Technologies

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq API
- OpenAI GPT-OSS 20B

## 📂 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
