from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingManager:
    """Handles document generation using Sentencse Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        Args:
            model_name: Huggingface model name for sentence embedding
        """
        
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"[DEBUG] Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"[DEBUG] Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"[ERROR] loading model {self.model_name}: {e}")
            
    def generate_embeddings(self, texts: List[str]):
        """
        Generate embeddings for a list of texts
        Args:
            texts: list of text strings to embed
        """
        
        if not self.model:
            print("[ERROR] model is not loaded")
            raise ValueError("Model not loaded")
        
        print(f"[DEBUG] Generating embedding for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[DEBUG] Generating embedding with shape: {embeddings.shape}")
        return embeddings
    

        
embedding_manager = EmbeddingManager()