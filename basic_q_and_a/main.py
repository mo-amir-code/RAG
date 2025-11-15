from src.ingest import load_all_documents
from src.chunker import split_documents
from src.embedder import embeddingManager
from src.util import docs_to_texts
from src.vector_store import vector_store 

def main():
    loaded_documents = load_all_documents("./data/pdf")
    chunks = split_documents(loaded_documents)
    
    # convert documents to texts
    texts = docs_to_texts(chunks)
    
    # Generate the embeddings
    embeddings = embeddingManager.generate_embeddings(texts)
    
    # Store in the vector database
    vector_store.add_documents(chunks, embeddings)    


if __name__ == "__main__":
    main()
