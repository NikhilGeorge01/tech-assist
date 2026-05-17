from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from tavily import TavilyClient


load_dotenv()


# -----------------------------
# Tavily Client
# -----------------------------
tavily = TavilyClient()


# -----------------------------
# State Schema
# -----------------------------
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retries: int
    hallucination_check: str


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

If the document is relevant to the question, respond ONLY with:
yes

Otherwise respond ONLY with:
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
            return "websearch"

        return "retry"

    return "generate"


# -----------------------------
# Query Rewrite Node
# -----------------------------
def rewrite_query(state: GraphState):

    print("\n---REWRITING QUERY---\n")

    question = state["question"]

    prompt = f"""
Rewrite this question in ONE short sentence for better retrieval.

Question:
{question}

Only return the rewritten query.
"""

    response = llm.invoke(prompt)

    better_question = response.content.strip()[:200]

    return {
        "question": better_question,
        "documents": [],
        "retries": state.get("retries", 0) + 1
    }


# -----------------------------
# Web Search Node
# -----------------------------
def web_search(state: GraphState):

    print("\n---WEB SEARCH FALLBACK---\n")

    question = state["question"][:200]

    response = tavily.search(
        query=question,
        max_results=3
    )

    results = response["results"]

    web_docs = []

    for result in results:

        content = result.get("content", "")

        web_docs.append(content)

    return {
        "question": question,
        "documents": web_docs,
        "retries": state.get("retries", 0)
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
# Hallucination Check Node
# -----------------------------
def check_hallucination(state: GraphState):

    print("\n---CHECKING HALLUCINATIONS---\n")

    question = state["question"]

    docs = state["documents"]

    generation = state["generation"]

    combined_docs = "\n\n".join(docs)

    prompt = f"""
You are a hallucination detection system.

Question:
{question}

Retrieved Context:
{combined_docs}

Generated Answer:
{generation}

Determine whether the generated answer is fully supported by the retrieved context.

Respond ONLY with:
yes

or

no
"""

    response = llm.invoke(prompt)

    result = response.content.strip().lower()

    return {
        "question": question,
        "documents": docs,
        "generation": generation,
        "retries": state.get("retries", 0),
        "hallucination_check": result
    }


# -----------------------------
# Hallucination Decision Node
# -----------------------------
def decide_hallucination(state: GraphState):

    print("\n---HALLUCINATION DECISION---\n")

    check = state["hallucination_check"]

    retries = state.get("retries", 0)

    if check == "yes":
        return "approved"

    if retries >= 2:
        return "approved"

    return "retry"


# -----------------------------
# Build Graph
# -----------------------------
graph_builder = StateGraph(GraphState)

graph_builder.add_node("retrieve", retrieve)

graph_builder.add_node("grade_documents", grade_documents)

graph_builder.add_node("rewrite_query", rewrite_query)

graph_builder.add_node("web_search", web_search)

graph_builder.add_node("generate", generate)

graph_builder.add_node("check_hallucination", check_hallucination)

graph_builder.set_entry_point("retrieve")

graph_builder.add_edge("retrieve", "grade_documents")

graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "retry": "rewrite_query",
        "websearch": "web_search"
    }
)

graph_builder.add_edge("rewrite_query", "retrieve")

graph_builder.add_edge("web_search", "generate")

graph_builder.add_edge("generate", "check_hallucination")

graph_builder.add_conditional_edges(
    "check_hallucination",
    decide_hallucination,
    {
        "approved": END,
        "retry": "rewrite_query"
    }
)

graph = graph_builder.compile()


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    result = graph.invoke({
        "question": "How does Kubernetes scheduling work?",
        "retries": 0
    })

    print("\nFINAL ANSWER:\n")

    print(result["generation"])