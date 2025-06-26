# 📄 Research Paper Q&A with LLMs

This repository enables semantic search and question-answering on research papers using PDF ingestion, vector databases, and local or cloud-hosted LLMs. It supports both OpenAI-compatible models (e.g., via LM Studio) and HuggingFace models like Falcon, integrating them into a LangChain-powered RAG pipeline.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Data Extraction](#data-extraction)
- [Data Transformation](#data-transformation)
- [Data Loading](#data-loading)
- [Visualization](#visualization)
- [Automation](#automation)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

This project allows users to query research paper PDFs in natural language. It processes the paper, stores chunk embeddings in a Chroma vector database, and retrieves relevant context at query time using LangChain and either OpenAI-compatible APIs (e.g., LM Studio) or HuggingFace models.

---

## 🏗️ Architecture

```
PDF -> LangChain Splitter -> Embeddings (OpenAI / HF) -> ChromaDB
          ↓                                         ↑
        Query  → Embed → Retrieve Similar Chunks → Prompt LLM → Answer
```

---

## 🧩 Components

- **LangChain**: Provides the orchestration layer for RAG.
- **ChromaDB**: Vector database to store and retrieve document embeddings.
- **HuggingFace / OpenAI Models**: Used for embedding and LLM completion.
- **LM Studio**: Used as a local server to host OpenAI-compatible models.

---

## 📥 Data Extraction

The system loads a research paper from `../data/paper1.pdf` using `PyPDFLoader`. Each page is treated as a LangChain `Document` and metadata is attached to track the source.

---

## 🧱 Data Transformation

The text is chunked into overlapping sections using `RecursiveCharacterTextSplitter`, then embedded using:

- `text-embedding-nomic-embed-text-v1.5` (via LM Studio API in `lmstudio.py`)
- or `HuggingFaceEmbeddings()` (in `main.py`)

---

## 💾 Data Loading

Embeddings are persisted in a Chroma vector store:

- `lmstudio.py` uses the raw Chroma Python client (`chromadb`) and you have to run a LLM and embedding model locally using LMStudio (I did on Windows)
- `main.py` uses `langchain_chroma.Chroma` with built-in LangChain support

Each document chunk is uniquely ID'd and tagged with its source metadata.

---

## 📊 Visualization

While the system is CLI-driven, it provides:

- Terminal display of matching document chunks
- Full prompts for debugging
- Grounding reference for every LLM-generated answer

---

## 🤖 Automation

Once set up, you can interactively query the ingested paper:

```bash
$ python main.py --llm mistral
Ask a question (or type 'exit'): What is the main contribution of the paper?
```

In `lmstudio.py`, the prompt is handcrafted and passed to the model via direct API calls to an OpenAI-compatible server[Which is running locally on LM Studio]

---

## ✅ Prerequisites

- Python 3.9+
- NVIDIA GPU (for local inference)
- [LM Studio](https://lmstudio.ai/) (optional for local inference)
- ChromaDB installed locally

---

## ⚙️ Setup

1. **Clone and Install Requirements**
   ```bash
   git clone https://github.com/your-username/research-paper-qa.git
   cd research-paper-qa
   pip install -r requirements.txt
   ```

2. **Add `.env` file**
   ```
   OPENAI_API_KEY=your-key-if-using-openai
   ```

3. **Download a PDF**
   Place your paper at: `../data/paper1.pdf`

4. **Optional: Start LM Studio**
   - Load a model like `gemma:7b` or `mistral`
   - Expose OpenAI-compatible API on `localhost:1234`

---

## 🚀 Usage

### Using Local LM Studio API

```bash
python lmstudio.py --llm openai
```

### Using HuggingFace Model Pipeline

```bash
python main.py --llm mistral
```

You’ll be prompted to enter your question. The system will embed it, perform retrieval, and generate an answer using the selected LLM.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
