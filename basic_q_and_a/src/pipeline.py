from .embedder import EmbeddingManager, embedding_manager
from .vector_store import VectorStore, vector_store
from typing import List, Dict

class RAGRetriever:
    """Handles query based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initializing the retriever
        Args:   
            vectore_store: VectorStore containing the documnets embeddings
            embedding_manager: Manager to generating query embedding
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents for a query
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
        """
        
        # Generate query embeddings
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k                
            )
            
            # Process results
            retrieved_docs = []
            
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]
                
                
                for i, (id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        }) 
                        
                print(f"[DEBUG] Retreived {len(retrieved_docs)} documents (after filtering)")
            else:
                print(f"[DEBUG] No documents found")
                
            return retrieved_docs
        except Exception as e:
            print(f"[ERROR] Occurred while searching in the vector store")  
            return retrieved_docs
        
        
rag_retriever = RAGRetriever(vector_store, embedding_manager)