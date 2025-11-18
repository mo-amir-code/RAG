from src.ingest import load_all_documents
from src.chunker import split_documents
from src.vector_store import create_vector_store
from src.generator import generator

class Setup:
    def __init__(self):
        documents = load_all_documents("./data/pdf")
        texts = split_documents(documents)
    
        retriever =  create_vector_store(texts, generator.embeddings)

        print(f"[DEBUG] Retriever has been created: {retriever}")
        
        self.documents = documents
        self.texts = texts
        self.retriever = retriever
        
        
setup = Setup()
