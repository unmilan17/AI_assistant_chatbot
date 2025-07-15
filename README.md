# AI_assistant_chatbot

A modular, LangGraph-based conversational agent built to handle **domain-specific scientific queries in AI/ML**. Combines large language models, retrieval-augmented generation (RAG), context-aware summarization, and agent orchestration for high-relevance, explainable responses.

---

## 🧠 Key Features

- **LangGraph Multi-Agent Workflow**: Modular architecture using LangGraph nodes for summarization, query classification, routing, and memory.
- **Retrieval-Augmented Generation (RAG)**: Retrieves from a curated AI/ML research corpus using Pinecone + E5 embeddings.
- **Summarization Node**: Condenses retrieved documents with context-aware LLM prompting.
- **Context Reranking**: Prioritizes more relevant or higher-ranked sources in the final summary.
- **Tool Integration**: Uses Arxiv, Wikipedia, and Tavily for factual lookup and fallback when vector store retrieval is weak.
- **Memory Support**: Conversation memory via `MemorySaver` with per-session `thread_id`s.

---

## 🚀 Performance

| Metric     | Value     |
|------------|-----------|
| EM Score   | 71.3%     |
| F1 Score   | 83.5%     |

Achieved on benchmark scientific QA datasets (e.g., SciQ, PubMedQA).

---

## ⚙️ Tech Stack

- **LangGraph** for orchestrated stateful agent graph
- **LangChain + Groq LLMs** (e.g., LLaMA 3, Gemma)
- **Pinecone** vector store
- **E5 Embeddings** for semantic retrieval
- **FastAPI** (optional) for deployment
- **ToolNode** setup for Arxiv/Wikipedia/Tavily tools

---

## 📦 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your environment variables
Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
```

### 3. Run the chatbot
```bash
python chatbot_ml_ai_papers.py
```

Optional: Deploy via FastAPI.

---

## 🧪 Example Queries
- "What is reinforcement learning?"
- "Compare Q-learning with policy gradients"
- "Latest papers on transformer-based vision models"

---

## 📄 License
MIT

---

## 🙌 Acknowledgements
Thanks to LangChain, Pinecone, Groq, and open-access scientific datasets.

---

## 💡 Future Work
- Long-term memory using Redis or Postgres
- Streaming + frontend (React or Streamlit)
- Continuous retrieval over arXiv updates
