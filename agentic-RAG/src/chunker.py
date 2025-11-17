from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size: int = 400, chunk_overlap: int = 80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    
    print(f"[DEBUG] Documents splitted successfully...") 
    print(f"[DEBUG] Documents: {len(split_docs)}") 
    
    return split_docs