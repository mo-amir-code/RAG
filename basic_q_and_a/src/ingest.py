from typing import List
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def load_all_documents(docs_dir: str) -> List[any]:
    """
    Loads all the supported files e.g., pdf
    """
    
    docs_path = Path(docs_dir).resolve()
    print(f"[DEBUG] Documents Path: {docs_path}")
    documents = []
    
    pdf_files = list(docs_path.glob("**/*.pdf"))
    # print(f"[DEBUG] FOUND {len(pdf_files)} PDF FILES, FILES: {[str(f) for f in pdf_files]}")
    print(f"[DEBUG] FOUND {len(pdf_files)} PDF FILES")
    
    for pdf_file in pdf_files:
        print(f"[DEBUG] Found {pdf_file} PDF file")
        
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR]: Error occurred while loading {pdf_file} PDF file")
            
    
    return documents
        