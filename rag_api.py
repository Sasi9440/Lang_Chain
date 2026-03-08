from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

app = FastAPI()

# ---- Enable CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load PDF ----
loader = PyPDFLoader("Econamics.pdf") # change if filename different
documents = loader.load()

# ---- Split Text ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# ---- Create Embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---- Use Local Model ----
llm = OllamaLLM(model="phi3:mini")  # or mistral if installed

# ---- Request Model ----
class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Strict PDF RAG API Running"}

# ---- Strict Context-Based Route ----
@app.post("/chat")
async def chat(q: Question):

    relevant_docs = retriever.invoke(q.question)

    if not relevant_docs:
        return {"answer": "Not found in document."}

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # STRICT prompt
    prompt = f"""
You are a strict document question answering system.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not clearly present in the context, reply exactly:
   Not found in document.
3. Do NOT use prior knowledge.
4. Do NOT explain.
5. Do NOT add extra text.

Context:
{context}

Question:
{q.question}

Answer:
"""

    response = llm.invoke(prompt).strip()

    # Extra safety filter
    if response == "" or "not found" in response.lower():
        return {"answer": "Not found in document."}

    return {"answer": response}