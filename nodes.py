import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load variables from .env immediately
load_dotenv()

# 2. Get the key and verify it exists
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file!")

# 3. Initialize the LLM with the explicit key
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    groq_api_key=api_key,
    temperature=0
)
def router_node(state):
    """Checks if the query needs a Human or AI."""
    q = state["query"].lower()
    critical_terms = ["refund", "legal", "sue", "scam", "unhappy"]
    
    if any(term in q for term in critical_terms):
        return {"next_action": "human"}
    return {"next_action": "ai"}

def rag_node(state, retriever):
    """Standard RAG logic."""
    context_docs = retriever.invoke(state["query"])
    context = "\n".join([d.page_content for d in context_docs])
    
    prompt = f"System: Use Context to answer. Context: {context}\nUser: {state['query']}"
    response = llm.invoke(prompt)
    return {"response": response.content}

def human_node(state):
    """Human-in-the-Loop Node."""
    print("\n--- [SYSTEM] ESCALATING TO HUMAN AGENT ---")
    human_input = input(f"Customer Query: {state['query']}\nExpert Response: ")
    return {"response": human_input}