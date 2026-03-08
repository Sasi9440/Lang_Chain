# Lang_Chain

A RAG-based question answering system using LangChain and FastAPI. Loads a PDF, splits content into chunks, converts them into embeddings, and stores them in a FAISS vector database. When a user asks a question, the system retrieves relevant context from the PDF and generates answers strictly based on the document content.

## Features
- **PDF Processing**: Loads and splits PDF documents into manageable chunks
- **Vector Embeddings**: Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model
- **Vector Database**: FAISS for efficient similarity search
- **LLM**: Ollama's `phi3:mini` model for local inference
- **Strict RAG**: Answers only from document context, no external knowledge

## Components

### main.py
Command-line interface for interactive Q&A with the PDF document.

### rag_api.py
FastAPI REST API with CORS support for web integration. Exposes `/chat` endpoint for question answering.

## Tech Stack
- LangChain Community
- FastAPI
- FAISS
- HuggingFace Embeddings
- Ollama (phi3:mini)

## 🚀 Setup and Running the Project

Before running the project, make sure Python and Ollama are installed on your system.

### 📦 Required Python Packages

```bash
pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers fastapi uvicorn pypdf ollama
```

### 📁 Project Structure
```
Lang_Chain
│
├── main.py
├── rag_api.py
├── Econamics.pdf
├── requirements.txt
├── README.md
└── venv
```

### ⚙️ Create Virtual Environment

```bash
python -m venv venv
```

### ▶️ Activate Virtual Environment
```bash
.\venv\Scripts\activate
```

When activated, your terminal will show:
```
(venv) PS D:\Lang_Chain>
```

### 📥 Install Required Packages
```bash
pip install -r requirements.txt
```

### 🤖 Install Ollama Model

Pull the local language model used for generating answers.

```bash
ollama pull phi3:mini
```

### ▶️ Run the CLI Application
```bash
python main.py
```

Example interaction:
```
Ask a question: What is scarcity?

Answer:
Scarcity refers to the limited availability of resources compared to unlimited human wants.
```

### 🌐 Run the FastAPI Server
```bash
uvicorn rag_api:app --reload
```

After running this command, the API server starts at: `http://127.0.0.1:8000`

### 🔗 API Endpoint

**POST /chat**

Request body:
```json
{
  "question": "What is opportunity cost?"
}
```

Response:
```json
{
  "answer": "Opportunity cost is the value of the next best alternative that must be forgone."
}
```

## 🔄 System Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline.

```
PDF Document
      │
      ▼
Text Extraction
      │
      ▼
Text Chunking
      │
      ▼
Embeddings Generation
      │
      ▼
FAISS Vector Database
      │
      ▼
User Question
      │
      ▼
Similarity Search
      │
      ▼
Relevant Context
      │
      ▼
Ollama LLM (phi3:mini)
      │
      ▼
Generated Answer
```

## 🎯 Key Features

- 📄 PDF-based question answering
- ⚡ Fast semantic search using FAISS
- 🧠 HuggingFace embedding model
- 🤖 Local LLM inference using Ollama
- 🌐 FastAPI integration for web applications
- 🔒 Answers generated strictly from document context

## 💡 Future Improvements

- Multiple document support
- Web chat interface
- Streaming responses
- Authentication for API
- Vector database persistence
