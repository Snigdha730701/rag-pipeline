# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict]



def load_documents(directory: str) -> List[str]:
    """Loads text documents from a specified directory."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            loader = TextLoader(filepath)
            documents.extend(loader.load())
    return documents