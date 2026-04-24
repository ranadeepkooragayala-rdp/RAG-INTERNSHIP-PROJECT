# 🤖 Agentic RAG: Customer Support Assistant
### Powered by LangGraph, ChromaDB, and Groq (Llama 3.3)

## 📌 Project Overview
This project is an **Intelligent Customer Support Assistant** that utilizes **Retrieval-Augmented Generation (RAG)** to provide accurate, document-grounded responses. Built with **LangGraph**, the system features an agentic workflow that can route queries based on intent and includes a **Human-in-the-Loop (HITL)** safety mechanism for high-risk escalations.

### Key Features
* **Fact-Grounded Responses:** Uses a PDF-based knowledge store to eliminate AI hallucinations.
* **Intent-Based Routing:** Distinguishes between general support and escalation-worthy queries (legal/refund threats).
* **Human-in-the-Loop:** Pauses the AI to allow manual intervention for sensitive topics.
* **Extreme Performance:** Powered by **Groq LPU** inference for sub-second response times.
* **Persistent Memory:** Local vector storage via **ChromaDB**.

---

## 🏗️ System Architecture
The project follows a modular design consisting of a document ingestion pipeline and a stateful query execution graph.

1.  **Ingestion:** PDF → Recursive Chunking → HuggingFace Embeddings → ChromaDB.
2.  **Inference:** User Query → Intent Router → (RAG Node OR Human Node) → Response.

---

## 🛠️ Tech Stack
* **LLM:** Llama 3.3 70B (via Groq API)
* **Orchestration:** LangGraph & LangChain
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Environment:** Python 3.10+, Dotenv

---

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone [https://github.com/yourusername/rag-support-bot.git](https://github.com/yourusername/rag-support-bot.git)
cd rag-support-bot

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Environment Setup
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
3. Run the Project
Place your policy PDF in the data/ folder, then run:

Bash
python main.py
📂 Project Structure
data/ - Contains the source PDF documents.

chroma_db/ - Persistent storage for vector embeddings.

ingestion.py - Script for processing PDFs and populating the vector store.

nodes.py - Contains the logic for Graph nodes and conditional routing.

main.py - The entry point that initializes and executes the LangGraph workflow.

requirements.txt - List of necessary Python packages.

📝 Design Decisions
Chunking: 1000 characters with 100 overlap to maintain context.

Retrieval: Top-3 similarity search to provide sufficient context to the LLM.

Routing: Keyword-based and semantic analysis to trigger human handoff for legal/financial risks.

👨‍💻 Author
Ranadeep Kooragayala 
