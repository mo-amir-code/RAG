from langchain_community.vectorstores import FAISS


def create_vector_store(docs, embeddings):
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(k=3)
    return retriever
    