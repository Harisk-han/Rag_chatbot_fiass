# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------
# 1. Load Environment Variables
# -------------------------
load_dotenv()  # Loads values from .env into environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# -------------------------
# 2. Load PDF
# -------------------------
loader = PyPDFLoader("hello.pdf")  # replace with your PDF
documents = loader.load()

# -------------------------
# 3. Split into Chunks
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# -------------------------
# 4. Embeddings
# -------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# 5. FAISS Index (save/load)
# -------------------------
faiss_index_path = "faiss_index"
if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(
        faiss_index_path, embeddings, allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# -------------------------
# 6. Gemini Flash LLM
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    convert_system_message_to_human=True,  # ⚠️ Will be deprecated soon
)

# -------------------------
# 7. Conversational Memory
# -------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff"
)

# -------------------------
# 8. FastAPI Setup
# -------------------------
app = FastAPI(title="PDF RAG Assistant API")


class QueryRequest(BaseModel):
    question: str


class BatchQueryRequest(BaseModel):
    questions: List[str]


@app.get("/")
def root():
    return {"message": "PDF RAG Assistant is running ✅"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    response = qa_chain.invoke({"question": request.question})
    return {"answer": response["answer"]}


@app.post("/ask_batch")
def ask_batch(request: BatchQueryRequest):
    results = []
    for q in request.questions:
        response = qa_chain.invoke({"question": q})
        results.append({"question": q, "answer": response["answer"]})
    return {"results": results}
