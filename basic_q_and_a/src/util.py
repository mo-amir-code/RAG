

def docs_to_texts(docs):
    texts = [doc.page_content for doc in docs]
    return texts

def retrieved_docs_to_text_only(docs):
    text = "\n\n".join(doc["content"] for doc in docs)
    return text