from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector store
vectorstore = Chroma(
    persist_directory="vectorstore",
    embedding_function=embedding_model
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test question
query = "What is FastAPI?"

# Retrieve documents
docs = retriever.invoke(query)

# Print results
for i, doc in enumerate(docs, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content)
    print("-" * 50)