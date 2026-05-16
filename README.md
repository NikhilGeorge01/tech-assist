# RAG-Based Technical Documentation Assistant

This project is a Retrieval-Augmented Generation (RAG) based technical documentation assistant built using LangGraph, FastAPI, ChromaDB, and Groq LLMs.

The system allows users to query technical documentation using natural language. It retrieves relevant document chunks from a vector database, grades their relevance using an LLM, and generates grounded answers using a LangGraph workflow.

## Features

- Document ingestion pipeline
- Markdown document loading
- Recursive text chunking
- Embedding generation using sentence-transformers
- ChromaDB vector storage
- Semantic document retrieval
- LangGraph workflow orchestration
- LLM-based document relevance grading
- Query rewriting and retry logic
- FastAPI backend
- Dynamic document ingestion endpoint
- Feedback endpoint
- Source-aware answer generation

## Tech Stack

- Python
- LangGraph
- LangChain
- FastAPI
- ChromaDB
- HuggingFace Embeddings
- Groq LLM API
- Sentence Transformers

## Project Structure

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

## Workflow Architecture

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
                           ↓
                     Retrieve Again
```

## Document Ingestion Pipeline

The ingestion pipeline performs the following steps:

1. Loads markdown documents from the `docs/` folder
2. Splits documents into smaller chunks using RecursiveCharacterTextSplitter
3. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
4. Stores embeddings inside ChromaDB

Run ingestion using:

```bash
python app/ingest.py
```

## Setup Instructions

### Clone Repository

```bash
git clone <your-repository-url>
cd tech-assist
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

## Running the Application

### Step 1: Ingest Documents

```bash
python app/ingest.py
```

### Step 2: Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

## API Endpoints

### POST `/query`

Submit a question to the RAG assistant.

Example request:

```json
{
  "question": "What is FastAPI?"
}
```

### GET `/documents`

Lists indexed documents from the corpus.

### POST `/feedback`

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

### POST `/ingest`

Upload and ingest new markdown documents dynamically.

## Design Decisions

### Why LangGraph

LangGraph was chosen because it allows stateful graph-based workflows with conditional routing. This makes it suitable for implementing self-corrective RAG pipelines.

### Why ChromaDB

ChromaDB was selected because it is lightweight, easy to set up locally, and integrates well with LangChain.

### Why HuggingFace Embeddings

The `all-MiniLM-L6-v2` embedding model was chosen because it is lightweight, fast, and sufficient for small-scale semantic retrieval tasks.

### Why Groq

Groq provides fast inference with free API access, making it suitable for rapid prototyping and experimentation.

## Tradeoffs

### Simplicity vs Accuracy

The project prioritizes simplicity and readability over production-level optimization. The current implementation is lightweight and easy to understand but may not perform as well on large-scale corpora.

### Small Embedding Model

A lightweight embedding model was used to reduce computational overhead and improve speed. Larger embedding models may improve retrieval quality at the cost of latency and resource usage.

### Basic Relevance Grading

Document grading currently uses a simple yes/no relevance classification. More advanced grading mechanisms or reranking systems could improve retrieval precision.

### Local Vector Database

ChromaDB runs locally for simplicity and ease of setup. A production deployment would likely use a managed vector database for scalability and persistence.

### Limited Retry Logic

The workflow currently uses a simple retry mechanism with query rewriting. More advanced adaptive retrieval strategies or web search fallback mechanisms could improve robustness.

## Future Improvements

- Add hallucination checking
- Add conversational memory
- Add web search fallback
- Add reranking models
- Add authentication and session handling
- Add frontend UI using Streamlit or React
- Add persistent feedback storage

## Example Query

Question:

```text
What is FastAPI?
```

Example Response:

```text
FastAPI is a modern Python web framework used for building APIs quickly and efficiently.
```
