from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()


# -----------------------------
# State Schema
# -----------------------------
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: int


# -----------------------------
# Load Vector Store
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -----------------------------
# Load LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant"
)


# -----------------------------
# Retrieval Node
# -----------------------------
def retrieve(state: GraphState):

    print("\n---RETRIEVING DOCUMENTS---\n")

    question = state["question"]

    docs = retriever.invoke(question)

    retrieved_docs = [doc.page_content for doc in docs]

    return {
        "question": question,
        "documents": retrieved_docs,
        "retries": state.get("retries", 0)
    }


# -----------------------------
# Document Grading Node
# -----------------------------
def grade_documents(state: GraphState):

    print("\n---GRADING DOCUMENTS---\n")

    question = state["question"]

    docs = state["documents"]

    relevant_docs = []

    for doc in docs:

        prompt = f"""
You are a document relevance grader.

Question:
{question}

Document:
{doc}

If the document is relevant to the question, respond only with:
yes

Otherwise respond only with:
no
"""

        response = llm.invoke(prompt)

        answer = response.content.strip().lower()

        if answer == "yes":
            relevant_docs.append(doc)

    return {
        "question": question,
        "documents": relevant_docs,
        "retries": state.get("retries", 0)
    }


# -----------------------------
# Decision Node
# -----------------------------
def decide_to_generate(state: GraphState):

    print("\n---DECISION NODE---\n")

    docs = state["documents"]

    retries = state.get("retries", 0)

    if len(docs) == 0:

        if retries >= 1:
            return "generate"

        return "retry"

    return "generate"


# -----------------------------
# Query Rewrite Node
# -----------------------------
def rewrite_query(state: GraphState):

    print("\n---REWRITING QUERY---\n")

    question = state["question"]

    prompt = f"""
Rewrite this question to improve document retrieval:

{question}
"""

    response = llm.invoke(prompt)

    better_question = response.content.strip()

    return {
        "question": better_question,
        "documents": [],
        "retries": state.get("retries", 0) + 1
    }


# -----------------------------
# Generation Node
# -----------------------------
def generate(state: GraphState):

    print("\n---GENERATING ANSWER---\n")

    question = state["question"]

    docs = state["documents"]

    combined_docs = "\n\n".join(docs)

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question ONLY using the provided context.

Context:
{combined_docs}

Question:
{question}

Answer clearly and accurately.

If the context is empty, say you do not have enough information.
"""

    response = llm.invoke(prompt)

    return {
        "question": question,
        "documents": docs,
        "generation": response.content,
        "retries": state.get("retries", 0)
    }


# -----------------------------
# Build Graph
# -----------------------------
graph_builder = StateGraph(GraphState)

graph_builder.add_node("retrieve", retrieve)

graph_builder.add_node("grade_documents", grade_documents)

graph_builder.add_node("rewrite_query", rewrite_query)

graph_builder.add_node("generate", generate)

graph_builder.set_entry_point("retrieve")

graph_builder.add_edge("retrieve", "grade_documents")

graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "retry": "rewrite_query"
    }
)

graph_builder.add_edge("rewrite_query", "retrieve")

graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    result = graph.invoke({
        "question": "What is FastAPI?",
        "retries": 0
    })

    print("\nFINAL ANSWER:\n")

    print(result["generation"])