import os
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables (e.g., for OLLAMA_HOST)
load_dotenv()

app = FastAPI(
    title="Local RAG Pipeline",
    description="A FastAPI endpoint for a local RAG pipeline using LangChain, Ollama, and FAISS.",
    version="1.0.0",
)

# --- Configuration ---
DOCUMENTS_DIR = "documents"
FAISS_DB_PATH = "faiss_index"
MODEL_NAME = "mistral" # Ensure this model is pulled via Ollama

# Get Ollama host from environment variable, default to localhost for local dev without Docker
# For Docker Desktop (macOS/Windows), use 'host.docker.internal'.
# For Linux, you might need '--network host' when running the container, or use the host's IP.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Global Variables for RAG components ---
vectorstore = None
llm = None
qa_chain = None
embeddings = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict]

# --- RAG Setup Functions ---

def load_documents(directory: str) -> List[str]:
    """Loads text documents from a specified directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    return documents

def initialize_rag_pipeline():
    """Initializes the RAG pipeline components: embeddings, LLM, and vectorstore."""
    global vectorstore, llm, qa_chain, embeddings

    print("Initializing RAG pipeline...")
    print(f"Attempting to connect to Ollama at: {OLLAMA_HOST}")

    # 1. Load Documents
    print(f"Loading documents from {DOCUMENTS_DIR}...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"No documents found in {DOCUMENTS_DIR}. Please add some .txt files.")
        raise RuntimeError(f"No documents found in {DOCUMENTS_DIR}. Cannot initialize RAG.")

    # 2. Split Documents
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 3. Initialize Embeddings (using OllamaEmbeddings)
    print(f"Initializing OllamaEmbeddings with model: {MODEL_NAME} and base_url: {OLLAMA_HOST}...")
    embeddings = OllamaEmbeddings(model=MODEL_NAME, base_url=OLLAMA_HOST)

    # 4. Create or Load FAISS Vector Store
    if os.path.exists(FAISS_DB_PATH):
        print(f"Loading existing FAISS index from {FAISS_DB_PATH}...")
        # IMPORTANT: allow_dangerous_deserialization=True is needed for loading FAISS indices
        # saved by langchain_community. Use with caution and only if you trust the source.
        vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore.save_local(FAISS_DB_PATH)
        print(f"FAISS index created and saved to {FAISS_DB_PATH}")

    # 5. Initialize LLM (using Ollama)
    print(f"Initializing Ollama LLM with model: {MODEL_NAME} and base_url: {OLLAMA_HOST}...")
    llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_HOST)

    # 6. Create RetrievalQA Chain
    print("Creating RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Other options: "map_reduce", "refine", "map_rerank"
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    print("RAG pipeline initialization complete.")


# --- FastAPI Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    """Event fired when the FastAPI application starts."""
    try:
        initialize_rag_pipeline()
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {e}")
        # Depending on criticality, you might want to exit or log more severely
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Event fired when the FastAPI application shuts down."""
    print("Shutting down RAG pipeline.")
    # Clean up resources if necessary (e.g., close database connections)


# --- FastAPI Endpoint ---

@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest):
    """
    Accepts a question, performs RAG using the local pipeline, and returns the answer
    along with source documents.
    """
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized. Please check server logs.")

    print(f"Received query: {request.question}")
    try:
        result = qa_chain.invoke({"query": request.question}) # LangChain 0.2.x uses invoke

        answer = result.get("result", "No answer found.")
        source_documents = result.get("source_documents", [])

        formatted_source_docs = []
        for doc in source_documents:
            # Metadata might contain 'source' which is the filename
            source_info = doc.metadata.get('source', 'N/A')
            formatted_source_docs.append({
                "page_content": doc.page_content,
                "metadata": {"source": source_info}
            })

        print(f"Answer: {answer}")
        return QueryResponse(answer=answer, source_documents=formatted_source_docs)

    except Exception as e:
        print(f"Error during RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

# This __name__ == "__main__" block is primarily for direct Python execution (e.g., `python main.py`)
# and is not typically used when running with Uvicorn via Docker.
if __name__ == "__main__":
    import uvicorn
    # When running directly, ensure Ollama server is running in the background.
    # The OLLAMA_HOST environment variable would need to be set before running this script
    # if Ollama is not on localhost:11434.
    uvicorn.run(app, host="0.0.0.0", port=8000)