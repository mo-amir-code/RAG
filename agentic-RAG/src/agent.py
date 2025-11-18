from typing import List, TypedDict
from langchain_core.documents import Document
from .generator import generator
from .setup import setup


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool


def decide_retrieval(state: AgentState) -> AgentState:
    """
    Decide if we need to retrieve documnets based on the question
    """
    question = state["question"]
    
    
    retrieval_keywords = ["what", "how", "explain", "describe", "tell me"]
    needs_retrieval = any(keyword in question.lower() for keyword in retrieval_keywords)

    return {**state, "needs_retrieval": needs_retrieval}


def retrieve_documents(state: AgentState) -> AgentState:
    """
    Retrieves relavent docoments based on the question
    """
    
    print(f"[DEBUG] Retriever: {setup.retriever}")
    
    question = state["question"]
    documents = setup.retriever.invoke(question)
    
    return {**state, "documents": documents}


def generate_answer(state: AgentState) -> AgentState:
    """
    Generate an answer using the retrieved documents or direct response
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if documents:
        # RAG approach: use documents as context
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:"""
    else:
        # Direct response without retrieval
        prompt = f"Answer the following question: {question}"
    
    response = generator.llm.invoke(prompt)
    answer = response.content
    
    return {**state, "answer": answer}


def should_retrieve(state: AgentState) -> str:
    """
    Determine the next step based on retrieval decision
    """
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"