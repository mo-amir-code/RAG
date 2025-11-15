from fastapi import FastAPI
from .pipeline import rag_retriever
from .util import retrieved_docs_to_text_only
from .generator import openai_llm
from fastapi.concurrency import run_in_threadpool


app = FastAPI()


@app.get("/health")
def health():
    return {"message": "App is running"}

@app.get("/rust/")
async def query(q:str):
    print(f"[DEBUG] query: {q}")

    # Run retriever in worker thread
    results = await run_in_threadpool(rag_retriever.retrieve, q)
    
    context = retrieved_docs_to_text_only(results)

    # Run openai LLM in worker thread
    response = await run_in_threadpool(openai_llm.generate_response, q, context)
    
    return {
        "message": "Query resolved",
        "data": response
    }