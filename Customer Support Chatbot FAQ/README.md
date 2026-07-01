# Customer Support Chatbot FAQ

A Retrieval-Augmented Generation (RAG) based FAQ assistant that indexes uploaded documents, creates a local vector store, and answers user questions grounded in the uploaded content.

## Project screenshot

![Customer Support Chatbot FAQ screenshot](screenshots/rag%20chatbot%20ss.png)

**Key features**

- Hybrid retrieval (fuzzy + semantic) over an indexed document corpus.
- Local FAISS vector store for embeddings.
- Streamlit frontend for uploading documents and interacting with the assistant.
- FastAPI backend that exposes an `/ask` endpoint for question answering.

**Quick links**

- Files: [main.py](Eccomerce Chatbot FAQ/main.py), [frontend.py](Customer Support Chatbot FAQ/frontend.py)
- Vector store builder: [build_vector_store.py](Customer Support Chatbot FAQ/build_vector_store.py)
- Document loader: [load_document.py](Customer Support Chatbot FAQ/load_document.py)
- RAG utilities: [rag_pipeline.py](Customer Support Chatbot FAQ/rag_pipeline.py)
- Search strategies: [searching/fuzzy.py](Customer Support Chatbot FAQ/searching/fuzzy.py), [searching/semantic.py](Customer Support Chatbot FAQ/searching/semantic.py), [searching/hybrid.py](Customer Support Chatbot FAQ/searching/hybrid.py)



Getting started
---------------

Prerequisites

- Python 3.10+ (or matching your environment)
- A virtual environment (recommended)

Install dependencies

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r "Customer Support Chatbot FAQ/requirements.txt"
```

Environment

- The project uses `.env` for configuration (see `main.py`). Add any API keys required by your LLM provider (OpenAI, Google GenAI, etc.) and MLflow tracking configuration if needed.

Build the vector store (index documents)

1. Put documents (PDF/DOCX/TXT) into the `Customer Support Chatbot FAQ/documents/` folder.
2. Run the builder to split and embed documents and save a local FAISS store:

```bash
python "Customer Support Chatbot FAQ/build_vector_store.py"
```

- The builder uses `sentence-transformers/all-MiniLM-L6-v2` by default and writes the index under `Customer Support Chatbot FAQ/vectorstore/db_faiss`.

Run the backend API

```bash
# from repository root
uvicorn "Customer Support Chatbot FAQ.main:app" --reload --host 0.0.0.0 --port 5021
```

- The backend exposes `POST /ask` which expects form data with a `text` field (the question). `main.py` loads the local FAISS store via `rag_pipeline.py` and uses the hybrid retriever in `searching/hybrid.py`.

Run the frontend (Streamlit)

```bash
streamlit run "Customer Support Chatbot FAQ/frontend.py" --server.port 8666
```

- The frontend allows file uploads (PDF/DOCX/TXT) and sending questions to the `/ask` endpoint. Note: the frontend in this repository is configured to call external URLs in the `UPLOAD_URL` / `ASK_URL` variables — update these to point at your local backend if you run it locally.

How it works (brief)

- Documents are loaded by `load_document.py`, split into chunks by `build_vector_store.py`, embedded with `HuggingFaceEmbeddings`, and stored in a FAISS index.
- `rag_pipeline.py` loads the saved FAISS index at startup.
- `main.py` receives a user question, uses `hybrid_langchain_retriever` to fetch context, formats a prompt and invokes an LLM (`ChatGoogleGenerativeAI` by default in the code).

Notes & recommendations

- The example uses `ChatGoogleGenerativeAI` / `ChatOpenAI` models — configure credentials and environment variables before running.
- The frontend references upload and ask endpoints on specific hosts/ports — change `UPLOAD_URL` and `ASK_URL` in `frontend.py` to your deployed backend.
- The `fuzzy_retriever` implementation is simplistic and may be adjusted for better matching behavior.
- For production, consider secure storage for keys, rate limiting, authentication, and model cost controls.

Troubleshooting

- If the vector store is not found, run `build_vector_store.py` to create `vectorstore/db_faiss`.
- Missing dependencies: ensure all packages from `requirements.txt` are installed in the active environment.
- If LLM invocation fails, check API keys and provider connectivity.

Contributing

- Bug reports and PRs welcome. Describe the environment used, steps to reproduce, and any logs.

---

Generated README stub for quick onboarding. Update secrets, endpoints, and any project-specific docs as needed.