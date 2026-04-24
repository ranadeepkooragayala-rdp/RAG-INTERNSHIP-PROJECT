import os
from dotenv import load_dotenv

# Run this FIRST
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, END
from ingestion import get_retriever
from nodes import router_node, rag_node, human_node
# Define State Schema
class AgentState(TypedDict):
    query: str
    response: str
    next_action: str

# 1. Setup Retriever
retriever = get_retriever()

# 2. Define the Graph
workflow = StateGraph(AgentState)

# 3. Add Nodes
workflow.add_node("intent_analyzer", router_node)
workflow.add_node("rag_engine", lambda state: rag_node(state, retriever))
workflow.add_node("human_expert", human_node)

# 4. Set entry and define edges
workflow.set_entry_point("intent_analyzer")

workflow.add_conditional_edges(
    "intent_analyzer",
    lambda x: x["next_action"],
    {
        "human": "human_expert",
        "ai": "rag_engine"
    }
)

workflow.add_edge("rag_engine", END)
workflow.add_edge("human_expert", END)

# 5. Compile and Run
app = workflow.compile()

if __name__ == "__main__":
    print("\nRAG Support Bot Active. (Type 'quit' to exit)")
    while True:
        user_msg = input("\nYou: ")
        if user_msg.lower() == "quit": break
        
        output = app.invoke({"query": user_msg})
        print(f"\nAssistant: {output['response']}")