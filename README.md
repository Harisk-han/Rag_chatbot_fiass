# 📄 PDF RAG Assistant API

A conversational question-answering API that lets you chat with any PDF document using Retrieval-Augmented Generation (RAG). Built with FastAPI, LangChain, FAISS, HuggingFace embeddings, and Google Gemini Flash.

---

## 🚀 Features

- Upload and index any PDF for semantic search
- Ask single or batched questions against your document
- Conversational memory — follow-up questions retain context
- Fast local vector search with FAISS (index is saved and reused)
- Powered by `gemini-1.5-flash` via Google Generative AI

---

## 🧱 Architecture

```
PDF → PyPDFLoader → RecursiveCharacterTextSplitter
    → HuggingFace Embeddings (all-MiniLM-L6-v2)
    → FAISS Vector Store
    → ConversationalRetrievalChain (Gemini Flash LLM + Memory)
    → FastAPI Endpoints
```

---

## 📋 Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. **Install dependencies**

```bash
pip install fastapi uvicorn python-dotenv langchain langchain-community \
    langchain-google-genai faiss-cpu sentence-transformers pypdf
```

3. **Set up your environment**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

4. **Add your PDF**

Place your PDF in the project root and rename it to `hello.pdf`, or update this line in `app.py`:

```python
loader = PyPDFLoader("hello.pdf")  # ← change to your filename
```

---

## ▶️ Running the API

```bash
uvicorn app:app --reload
```

The server will start at `http://127.0.0.1:8000`.

> **Note:** The FAISS index is built on the first run and saved to `./faiss_index/`. Subsequent starts will load it from disk, making startup significantly faster.

---

## 📡 API Endpoints

### `GET /`
Health check.

**Response:**
```json
{ "message": "PDF RAG Assistant is running ✅" }
```

---

### `POST /ask`
Ask a single question about the PDF. Conversation history is retained between calls.

**Request body:**
```json
{ "question": "What is the main topic of the document?" }
```

**Response:**
```json
{ "answer": "The document discusses..." }
```

---

### `POST /ask_batch`
Ask multiple questions in a single request.

**Request body:**
```json
{
  "questions": [
    "What are the key findings?",
    "Who are the authors?"
  ]
}
```

**Response:**
```json
{
  "results": [
    { "question": "What are the key findings?", "answer": "..." },
    { "question": "Who are the authors?", "answer": "..." }
  ]
}
```

---

## 🧪 Testing with cURL

```bash
# Health check
curl http://127.0.0.1:8000/

# Single question
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize this document."}'

# Batch questions
curl -X POST http://127.0.0.1:8000/ask_batch \
  -H "Content-Type: application/json" \
  -d '{"questions": ["What is this about?", "List the main points."]}'
```

You can also use the interactive docs at `http://127.0.0.1:8000/docs`.

---

## 📁 Project Structure

```
.
├── app.py              # Main FastAPI application
├── hello.pdf           # Your source PDF (user-supplied)
├── faiss_index/        # Auto-generated FAISS vector index
├── .env                # API keys (not committed to git)
└── README.md
```

---

## ⚠️ Notes

- **Conversational memory** is stored in-process. Restarting the server resets the conversation history.
- The `convert_system_message_to_human=True` flag in the Gemini LLM config is marked for deprecation by LangChain — watch for updates.
- For large PDFs, the initial indexing may take a moment. The saved FAISS index avoids this on subsequent runs.

---

## 📄 License

MIT
