import os
import chromadb
from typing import List
import numpy as np
import uuid

class VectorStore:
    """Manages documents embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the chromeDB collection
            persist_directory: Directory to persist the vector store
        """
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
        
    def _initialize_store(self):
        """
        Initialize the ChromaDB client and collection
        """
        try:
            # Create persistent chromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF documents embedding for RAG"}
            )
            
            print(f"[DEBUG] Vector store initialized. Collection: {self.collection_name}")
            print(f"[DEBUG] Exsiting documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"[ERROR] Occurred while initializing vector store: {e}")
            raise
        
    def add_documents(self, documents: List[any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match with number of embeddings")
        
        print(f"[DEBUG] Adding {len(documents)} documents to vector store")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (document, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique id
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            
            # Prepare metadata
            metadata = dict(document.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(document.page_content)
            metadatas.append(metadata)

            # Prepare content
            documents_text.append(document.page_content)
            
            # Embeddings 
            embeddings_list.append(embedding.tolist())
            
            
        # Add to colletion
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            
            print(f"[DEBUG] Successfully added {len(documents)} documents to vector store")
            print(f"[DEBUG] Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"[ERROR] Occurred while adding documents in vector store: {e}")
            raise
        

vector_store = VectorStore()