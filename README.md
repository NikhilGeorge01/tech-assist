# RAG-Based Technical Documentation Assistant

This project is a Retrieval-Augmented Generation (RAG) based technical documentation assistant built using LangGraph, FastAPI, ChromaDB, Groq LLMs, and Tavily Search.

The system answers technical questions by retrieving relevant documentation chunks from a vector database, grading their relevance using an LLM, generating grounded answers, validating generated responses for hallucinations, and falling back to web search when local retrieval fails.

The project implements a self-corrective and self-reflective RAG workflow inspired by Self-RAG and Corrective RAG (CRAG).

---

# Features

- Document ingestion pipeline
- Markdown document loading
- Recursive text chunking
- Embedding generation using sentence-transformers
- ChromaDB vector storage
- Semantic retrieval
- LangGraph workflow orchestration
- LLM-based document relevance grading
- Query rewriting and retry logic
- Hallucination detection node
- Tavily web-search fallback
- FastAPI backend
- Dynamic document ingestion endpoint
- Feedback endpoint
- Source-grounded answer generation

---

# Tech Stack

- Python
- LangGraph
- LangChain
- FastAPI
- ChromaDB
- HuggingFace Embeddings
- Groq LLM API
- Tavily Search API
- Sentence Transformers

---

# Project Structure

```text
tech-assist/
│
├── app/
│   ├── ingest.py
│   ├── graph.py
│   ├── main.py
│
├── docs/
│
├── vectorstore/
│
├── requirements.txt
├── README.md
└── .env
```

---

# Workflow Architecture

```text
User Query
    ↓
Retrieve Documents
    ↓
Grade Retrieved Documents
    ↓
Relevant Documents Found?
   /                    \
 Yes                     No
 ↓                        ↓
Generate Answer       Rewrite Query
 ↓                        ↓
Hallucination Check   Retrieve Again
 ↓                        ↓
Approved?           Still No Results?
 /     \                  /      \
Yes     No             Yes       No
 ↓       ↓              ↓         ↓
Return   Retry      Web Search   Retry
            ↓            ↓
        Retrieve      Generate
```

---

# System Components

## 1. Document Ingestion Pipeline

The ingestion pipeline performs the following steps:

1. Loads markdown documents from the `docs/` folder
2. Splits documents into smaller chunks using RecursiveCharacterTextSplitter
3. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
4. Stores embeddings inside ChromaDB

Run ingestion using:

```bash
python app/ingest.py
```

---

## 2. Retrieval System

The system converts user queries into embeddings and retrieves semantically similar chunks from ChromaDB.

Retriever behavior includes:
- semantic similarity search
- top-k retrieval
- document filtering through relevance grading

---

## 3. Self-Corrective LangGraph Workflow

The system uses LangGraph StateGraph to implement a graph-based RAG workflow.

The workflow includes:
- retrieval node
- document grading node
- query rewriting node
- web-search fallback node
- generation node
- hallucination checking node

Conditional routing is used to dynamically decide the next step based on retrieval quality and hallucination verification.

---

# Hallucination Checking

The system includes a hallucination detection node inspired by Self-RAG.

After answer generation, the generated response is verified against the retrieved context using an LLM.

If the answer is not sufficiently grounded in the retrieved context:
- the workflow retries retrieval
- or falls back to web search

This improves answer reliability and reduces unsupported responses.

---

# Web Search Fallback

If the vector store cannot retrieve sufficiently relevant documents, the workflow falls back to Tavily web search.

The retrieved web results are then used as additional context for answer generation.

This improves robustness for:
- unseen queries
- out-of-domain questions
- incomplete local documentation corpora

---

# Setup Instructions

## Clone Repository

```bash
git clone <your-repository-url>
cd tech-assist
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
TAVILY_API_KEY=your_api_key_here
```

---

# Running the Application

## Step 1: Ingest Documents

```bash
python app/ingest.py
```

---

## Step 2: Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

---

## Step 3: Open Swagger UI

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## POST `/query`

Submit a question to the RAG assistant.

Example request:

```json
{
  "question": "What is FastAPI?"
}
```

---

## GET `/documents`

Lists indexed documents from the corpus.

---

## POST `/feedback`

Submit feedback on generated answers.

Example request:

```json
{
  "question": "What is FastAPI?",
  "answer": "FastAPI is a Python framework",
  "rating": "thumbs_up",
  "comment": "Good answer"
}
```

---

## POST `/ingest`

Upload and ingest new markdown documents dynamically.

---

# Design Decisions

## Why LangGraph

LangGraph was chosen because it allows stateful graph-based workflows with conditional routing. This makes it suitable for implementing self-corrective and adaptive RAG pipelines.

---

## Why ChromaDB

ChromaDB was selected because it is lightweight, easy to set up locally, and integrates well with LangChain.

---

## Why HuggingFace Embeddings

The `all-MiniLM-L6-v2` embedding model was chosen because it is lightweight, fast, and sufficient for small-scale semantic retrieval tasks.

---

## Why Groq

Groq provides fast inference with free API access, making it suitable for rapid prototyping and experimentation.

---

## Why Tavily

Tavily was selected because it is optimized for AI workflows and provides clean web-search results suitable for RAG systems.

---

# Tradeoffs

## Simplicity vs Accuracy

The project prioritizes simplicity, readability, and modularity over production-scale optimization. The current implementation is lightweight and easy to understand but may not perform optimally on extremely large corpora.

---

## Lightweight Embedding Model

A smaller embedding model was used to reduce computational overhead and improve speed. Larger embedding models could improve retrieval accuracy at the cost of higher latency and memory usage.

---

## Basic Relevance Grading

Document grading currently uses a binary yes/no classification. More advanced reranking models or confidence-based retrieval systems could improve precision.

---

## LLM-Based Hallucination Checking

Hallucination detection relies on another LLM evaluation step. While useful, this approach may still occasionally misclassify grounded or hallucinated responses.

---

## Local Vector Database

ChromaDB runs locally for simplicity and ease of setup. A production deployment would likely use a distributed vector database for scalability and persistence.

---

## Limited Retry Strategy

The workflow currently uses a simple retry and rewrite strategy. More advanced adaptive retrieval planning could improve robustness further.

---

## Web Search Dependency

Web-search fallback improves robustness but introduces external API dependency, latency, and potential variability in retrieved results.

---

# Future Improvements

- Add conversational memory
- Add reranking models
- Add citation extraction
- Add streaming responses
- Add authentication and session handling
- Add frontend UI using Streamlit or React
- Add persistent feedback storage
- Add multi-document source attribution
- Add agentic tool-use capabilities
- Add caching for retrieval and web search

---

# Example Queries

```text
What is FastAPI?
```

```text
How does Kubernetes scheduling work?
```

---

# Example Response

```text
FastAPI is a modern Python web framework used for building APIs quickly and efficiently.
```

---

# Author

Nikhil George
