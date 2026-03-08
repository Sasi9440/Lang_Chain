from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# -----------------------------
# LOAD PDF (make sure filename matches exactly)
# -----------------------------
loader = PyPDFLoader("Econamics.pdf") # change if spelling is different
documents = loader.load()

# -----------------------------
# SPLIT DOCUMENT INTO CHUNKS
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# -----------------------------
# CREATE EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# STORE IN FAISS
# -----------------------------
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# CONNECT TO LOCAL OLLAMA MODEL
# -----------------------------
llm = Ollama(model="phi3:mini")  # use mistral if installed

print("PDF loaded successfully. Ask your questions.\n")

# -----------------------------
# CHAT LOOP
# -----------------------------
while True:
    query = input("Enter your question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    relevant_docs = retriever.invoke(query)

    if not relevant_docs:
        print("\nAnswer: Not found in document.\n")
        continue

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
You are a strict document question answering system.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not clearly present, reply exactly:
   Not found in document.
3. Do NOT use outside knowledge.
4. Do NOT explain.
5. Do NOT add extra text.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt).strip()

    # Extra safety check
    if response == "" or "not found" in response.lower():
        print("\nAnswer: Not found in document.\n")
    else:
        print("\nAnswer:\n", response, "\n")