# 📄 RAG PDF Question Answering System

**RAG Project** is a Retrieval-Augmented Generation (RAG) based system that enables users to ask questions from PDF documents and receive accurate, AI-generated answers. It combines document ingestion, vector embeddings, semantic search, and large language models to provide context-aware responses.

This repository contains the complete backend pipeline for document loading, indexing, retrieval, and answer generation.

---

## 🚀 Features

* **PDF Document Ingestion**: Automatically loads and processes PDF files.
* **Text Chunking**: Splits large documents into manageable chunks.
* **Vector Embeddings**: Generates embeddings using HuggingFace models.
* **FAISS Vector Store**: Stores and retrieves document vectors efficiently.
* **Semantic Search**: Retrieves relevant document sections based on queries.
* **LLM Integration**: Uses Groq LLM for answer generation.
* **JSON Output Support**: Saves results in structured JSON format.
* **Modular Architecture**: Easy to extend and maintain.

---

## 🛠️ Tech Stack

* **Programming Language**: Python 3.10+
* **Embeddings**: HuggingFace Transformers
* **Vector Database**: FAISS
* **LLM Provider**: Groq
* **Framework**: LangChain
* **File Processing**: PyPDF
* **Environment Management**: Conda / venv

---

## 📋 Prerequisites

Make sure you have the following installed:

* Python 3.10 or higher
* Git
* Conda / Virtual Environment (recommended)
* Groq API Key

---

## ⚙️ Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Poojaprakash21/RAG_PROJECT.git
cd RAG_PROJECT
```

---

### 2️⃣ Create and Activate Virtual Environment (Optional but Recommended)

#### Using Conda

```bash
conda create -n rag_env python=3.10
conda activate rag_env
```

#### Using venv

```bash
python -m venv rag_env
rag_env\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Make sure to replace `your_groq_api_key_here` with your actual API key.

---

### 5️⃣ Add PDF Files

Place your PDF documents inside the `research_papers/` folder.

Example:

```
research_papers/
├── paper1.pdf
├── paper2.pdf
```

---

### 6️⃣ Run the Pipeline

To process documents and start querying:

```bash
python pipeline.py
```

Or run the application:

```bash
python app.py
```

---

## 📂 Project Structure

```
RAG_PROJECT/
├── embeddings/
│   └── embedder.py       # Generates vector embeddings
├── ingestion/
│   ├── loader.py         # Loads PDF files
│   └── chunker.py        # Splits text into chunks
├── llm/
│   └── groq_llm.py       # Groq LLM integration
├── retrieval/
│   └── retriever.py      # Semantic search logic
├── utils/
│   ├── json_writer.py    # JSON output handler
│   └── pdf_grouping.py   # PDF grouping utilities
├── vectorstore/
│   └── faiss_store.py    # FAISS vector database
├── research_papers/      # Input PDF files
├── app.py                # Application entry point
├── pipeline.py           # Main RAG pipeline
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md
```

---

## 🔄 Workflow Overview

1. Load PDF documents
2. Split text into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Send context to LLM
7. Generate final answer

---

## 📌 Example Usage

Once the system is running, you can ask questions like:

```
What is the main contribution of the research paper?
Explain the methodology used in section 3.
Summarize the conclusion.
```

The system will retrieve relevant content and generate an AI-based response.

---


Just tell me 👍
