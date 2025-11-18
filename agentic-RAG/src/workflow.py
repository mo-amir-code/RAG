from langgraph.graph import StateGraph, END
from .agent import AgentState, decide_retrieval, retrieve_documents, generate_answer, should_retrieve

class Workflow:
    
    def __init__(self):
            # Create the state graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("decide", decide_retrieval)
        workflow.add_node("retrieve", retrieve_documents)
        workflow.add_node("generate", generate_answer)

        # Set entry point
        workflow.set_entry_point("decide")

        # Add conditional edges
        workflow.add_conditional_edges(
            "decide",
            should_retrieve,
            {
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )

        # Add edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile the graph
        app = workflow.compile()
        self.app = app
        
        
        
workflow = Workflow()