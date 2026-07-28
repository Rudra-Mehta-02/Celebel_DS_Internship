# Retrieval-Augmented Generation (RAG) System

(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<Rudra-Mehta-02>/<Celebel_DS_Internship
>/blob/main/Week7_Rudra_Mehta_DPGU.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A simple, end-to-end Retrieval-Augmented Generation (RAG) pipeline that answers questions from a custom PDF document by combining semantic retrieval with a language model. Built as a Week 7 assignment.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Steps](#pipeline-steps)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Parameters](#key-parameters)
- [Example](#example)
- [Running Locally vs Google Colab](#running-locally-vs-google-colab)
- [Limitations](#limitations)
- [Applications](#applications)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Author](#author)

---

## Overview

This project implements a complete RAG pipeline that:
1. Loads a PDF document
2. Splits it into overlapping text chunks
3. Converts chunks into vector embeddings
4. Stores embeddings in a FAISS vector database
5. Retrieves the most relevant chunks for a user's query
6. Generates a context-grounded answer using an LLM (TinyLlama)

The system answers strictly from the retrieved document content, reducing hallucinations and improving factual accuracy compared to relying on the LLM's pre-trained knowledge alone.

---

## Architecture

```
PDF Document → Text Splitter → Embeddings → FAISS Vector DB
                                                    │
User Query → Query Embedding → Similarity Search ──┘
                                        │
                              Retrieved Chunks (top-k)
                                        │
                        Context + Query → Prompt → TinyLlama → Answer
```

---

## Pipeline Steps

| Step | Description | Tool/Model |
|------|-------------|------------|
| 1. Document Ingestion | Load PDF/text files | `PyPDFLoader` |
| 2. Text Chunking | Split text into overlapping chunks | `RecursiveCharacterTextSplitter` |
| 3. Text Embedding | Convert chunks into 384-dim vectors | `sentence-transformers/all-MiniLM-L6-v2` |
| 4. Vector Database | Index embeddings for similarity search | `FAISS` |
| 5. Load LLM | Load a pre-trained language model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| 6. Query Processing | Convert user query into an embedding | Same embedding model as Step 3 |
| 7. Context Retrieval | Retrieve top-k most relevant chunks | FAISS similarity search |
| 8. Create Context | Concatenate retrieved chunks | Plain Python string join |
| 9. Create Prompt | Build a context-grounded prompt | Custom prompt template |
| 10. Answer Generation | Generate the final answer | HuggingFace `pipeline` |

---

## Tech Stack

- **Python**
- **LangChain** (`langchain`, `langchain-community`, `langchain-huggingface`)
- **FAISS** (CPU) — vector similarity search
- **HuggingFace Transformers** — model pipeline
- **Sentence-Transformers** — embedding generation
- **PyPDF** — PDF parsing
- **Google Colab** — development/runtime environment

---

## Repository Structure

```
├── Week7_Rudra_Mehta_DPGU   # Main RAG pipeline notebook
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<Rudra-Mehta-02>/<Celebel_DS_Internship
>.git
cd <Celebal_DS_Internship>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Open `Week7_Rudra_Mehta_DPGU` in Google Colab (or Jupyter).
2. Run all cells to install dependencies and load the embedding + language models.
3. Upload your PDF document when prompted.
4. Enter your question when prompted:
   ```
   Ask your question: <your question here>
   ```
5. The system retrieves the most relevant chunks from the document and generates an answer grounded strictly in that content.

---

## Key Parameters

| Parameter | Value |
|---|---|
| Chunk size | 400 characters |
| Chunk overlap | 50 characters |
| Embedding model | `all-MiniLM-L6-v2` (384 dimensions) |
| Retrieval top-k | 8 |
| LLM | `TinyLlama-1.1B-Chat-v1.0` |
| max_new_tokens | 300 |
| Temperature | 0.1 |
| top_p | 0.95 |

---

## Example

**Question:**
```
What is Retrieval-Augmented Generation?
```

**Answer:**
```
(Generated from the content of the uploaded PDF. If the answer is not
present in the document, the system responds:
"Answer not found in the document.")
```

---

## Running Locally vs Google Colab

This notebook was originally built for **Google Colab** and uses `google.colab.files.upload()` to upload the PDF interactively.

To run it **locally** instead:
- Replace the `google.colab.files.upload()` cell with a hardcoded file path, e.g.:
  ```python
  file_path = "./your_document.pdf"
  ```
- Ensure you have a GPU (recommended) or sufficient CPU/RAM, since `TinyLlama-1.1B` and the embedding model will run locally.

---

## Limitations

- Uses a lightweight LLM (`TinyLlama-1.1B`) — good for demonstration, not production-grade accuracy.
- Answers are strictly limited to the retrieved context; the model will not use outside knowledge.
- No persistence — the FAISS index is rebuilt every time the notebook runs.
- Designed and tested primarily for Google Colab.

---

## Applications

- Document question answering
- Enterprise knowledge management
- Intelligent search systems
- Research assistants
- AI-powered chatbots

---

## Future Improvements

- [ ] Support multiple document uploads (multi-PDF knowledge base)
- [ ] Add a UI using Streamlit or Gradio
- [ ] Swap TinyLlama for a larger, more capable LLM
- [ ] Persist the FAISS index to disk instead of rebuilding on every run
- [ ] Add evaluation metrics (faithfulness, relevance, answer quality)
- [ ] Support additional file formats (DOCX, TXT, HTML)

---

## Acknowledgments

- Built as part of a Week 7 course assignment.
- Powered by [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and [HuggingFace](https://huggingface.co/).

---

## Author

**Rudra Mehta**
Week 7 Assignment — Retrieval-Augmented Generation (RAG) System
