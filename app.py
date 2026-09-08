import os
import hashlib
from typing import List, Dict

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    .source-box {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 8px;
        margin-top: 8px;
        border-left: 4px solid #4CAF50;
    }

    .answer-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(pdf_file) -> List[Dict]:
    """
    Extract text from each page of a PDF.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    pdf_bytes = pdf_file.getvalue()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    document.close()

    return pages


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[Dict]:

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        # Normalize whitespace
        text = " ".join(text.split())

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            if end >= len(text):
                break

            start = end - overlap

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def build_faiss_index(chunks: List[Dict]):

    embedding_model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # Inner Product + normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# SEARCH DOCUMENT
# ============================================================

def search_documents(
    query: str,
    index,
    chunks: List[Dict],
    top_k: int = TOP_K
):

    embedding_model = load_embedding_model()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        results.append(
            {
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    if "GROQ_API_KEY" not in st.secrets:

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )

        st.stop()

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict]
):

    client = get_groq_client()

    context_parts = []

    for i, result in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {result['page']}

{result['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions using ONLY the provided HR Policy
document context.

Rules:

1. Use only information contained in the supplied context.
2. Do not invent HR policies.
3. If the answer cannot be found in the context, clearly say:
   "I could not find this information in the uploaded HR policy."
4. Give a concise but useful answer.
5. When possible, mention the relevant policy section or page.
6. Do not treat your answer as legal advice.
7. If the question is ambiguous, explain what is unclear.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question based strictly on the HR policy context.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1500,
        reasoning_effort="low"
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 HR Policy Document")

    uploaded_file = st.file_uploader(
        "Upload an HR Policy PDF",
        type=["pdf"]
    )

    st.divider()

    st.markdown(
        """
        ### How it works

        1. Upload your HR Policy PDF.
        2. The PDF is converted into text.
        3. Text is divided into chunks.
        4. Sentence Transformers creates embeddings.
        5. FAISS finds relevant sections.
        6. Groq GPT-OSS 20B generates the answer.

        ### Technology

        - Streamlit
        - PyMuPDF
        - Sentence Transformers
        - FAISS
        - Groq
        - GPT-OSS 20B
        """
    )


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-title">📚 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your uploaded HR Policy document using RAG.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(pdf_bytes).hexdigest()

    # Only process when a new document is uploaded
    if current_hash != st.session_state.document_hash:

        with st.spinner(
            "Processing HR Policy PDF..."
        ):

            try:

                pages = extract_pdf_text(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.info(
                        "If this is a scanned PDF, OCR may be required."
                    )

                    st.stop()

                chunks = create_chunks(pages)

                if not chunks:

                    st.error(
                        "Could not create text chunks from the PDF."
                    )

                    st.stop()

                index = build_faiss_index(chunks)

                st.session_state.document_hash = current_hash
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.messages = []

                st.success(
                    f"PDF processed successfully: "
                    f"{len(pages)} pages, "
                    f"{len(chunks)} chunks."
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {str(e)}"
                )

    else:

        st.success(
            f"Document ready: "
            f"{len(st.session_state.chunks)} chunks indexed."
        )


# ============================================================
# CHAT INTERFACE
# ============================================================

if st.session_state.index is None:

    st.info(
        "👈 Upload an HR Policy PDF from the sidebar to begin."
    )

else:

    st.subheader("💬 Ask a Question")

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📌 View sources"
                ):

                    for source in message["sources"]:

                        st.markdown(
                            f"""
                            **Page {source['page']}**
                            
                            > {source['text']}
                            
                            Similarity: `{source['score']:.3f}`
                            """
                        )

    question = st.chat_input(
        "Example: How many annual leave days are allowed?"
    )

    if question:

        # Show user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Retrieve relevant chunks
        with st.spinner(
            "Searching the HR policy..."
        ):

            retrieved_chunks = search_documents(
                question,
                st.session_state.index,
                st.session_state.chunks,
                TOP_K
            )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                    st.markdown(
                        f'<div class="answer-box">'
                        f'{answer}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    with st.expander(
                        "📌 View sources"
                    ):

                        for source in retrieved_chunks:

                            st.markdown(
                                f"""
                                **Page {source['page']}**

                                > {source['text']}

                                Similarity:
                                `{source['score']:.3f}`
                                """
                            )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Groq API error: {str(e)}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HR Policy Assistant • RAG + FAISS + Sentence Transformers + Groq"
)
