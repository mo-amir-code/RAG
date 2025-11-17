from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from pathlib import Path

def load_all_documents(docs_dir:str = "./data/pdf") -> List[any]:
    """Loades all the pdf files from the provided directory"""

    docs_resolved_path = Path(docs_dir).resolve()
    
    documents = []
    
    pdf_files = list(docs_resolved_path.glob("**/*.pdf"))
    
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading {pdf_file} PDF File...")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            loaded = loader.load()
            documents.extend(loaded)
            print(f"[DEBUG] {pdf_file} file text has been loaded")
        except Exception as e:
            print(f"[ERROR] Occurred while loading pdf file: {pdf_file}")


    return documents