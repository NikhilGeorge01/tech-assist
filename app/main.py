from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File
import shutil
import subprocess
from app.graph import graph


app = FastAPI()


# -----------------------------
# Request Schema
# -----------------------------
class QueryRequest(BaseModel):
    question: str


# -----------------------------
# Root Route
# -----------------------------
@app.get("/")
def root():
    return {"message": "RAG Assistant Running"}


# -----------------------------
# Query Route
# -----------------------------
@app.post("/query")
def query_rag(request: QueryRequest):

    result = graph.invoke({
        "question": request.question,
        "retries": 0
    })

    return {
        "question": request.question,
        "answer": result["generation"]
    }
@app.get("/documents")
def list_documents():

    import os

    files = os.listdir("docs")

    return {
        "documents": files
    }
class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: str = ""

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):

    print("\n---FEEDBACK RECEIVED---")
    print(feedback)

    return {
        "message": "Feedback received successfully"
    }

@app.post("/ingest")
def ingest_document(file: UploadFile = File(...)):

    file_path = f"docs/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Re-run ingestion pipeline
    subprocess.run(["python", "app/ingest.py"])

    return {
        "message": f"{file.filename} ingested successfully"
    }