from dotenv import load_dotenv
load_dotenv()
from src.ingest import load_all_documents
from src.chunker import split_documents
from src.embedder import embedding_manager
from src.util import docs_to_texts, retrieved_docs_to_text_only
from src.vector_store import vector_store 
from src.pipeline import rag_retriever
from src.generator import openai_llm

def main():
    loaded_documents = load_all_documents("./data/pdf")
    chunks = split_documents(loaded_documents)
    
    # convert documents to texts
    texts = docs_to_texts(chunks)
    
    # Generate the embeddings
    embeddings = embedding_manager.generate_embeddings(texts)
    
    # Store in the vector database
    vector_store.add_documents(chunks, embeddings)    
    
    ask_user()


def ask_user():
    while True:
        try:
            query = input("Enter query: ")
            
            results = rag_retriever.retrieve(query)
            context = retrieved_docs_to_text_only(results)
            
            response = openai_llm.generate_response(query, context)
            
            print(f"[DEBUG] RESPONSE: {response} \n\n\n")            
            
        except Exception as e:
            print(f"[ERROR] occurred while retrieving")

if __name__ == "__main__":
    main()
