from src.setup import setup
from src.workflow import workflow

def main():
    while True:
        query = input("Enter query: ")
        ask_question(query)
    
    
    
def ask_question(question: str):
    """
    Helper function to ask a question and get an answer
    """
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False
    }
    
    result = workflow.app.invoke(
        initial_state
    )
    
    print(f"[DEBUG] Result: {result["answer"]}")


if __name__ == "__main__":
    main()
