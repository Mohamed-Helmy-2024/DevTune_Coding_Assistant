# DevTune — Django + FastAPI RAG-powered assistant

DevTune is a full-stack example project providing an AI-based chat and RAG (Retrieval-Augmented Generation) system.
It uses a Django app for the frontend and user management and a lightweight FastAPI service that handles embedding generation, vector storage, and RAG search.

Key features:

- Per-user file uploads with per-user vector indexes
- Automatic indexing of uploaded files into a vector store (FAISS/Chroma/in-memory)
- Deletion of documents removes vectors from the vector database
- Configurable LLM provider(s) for both generation and embedding (OpenAI, Cohere, Ollama)
- Optional RAG context in chat conversations via a simple checkbox in the UI
- Frontend and backend sync: Django persists `KnowledgeFile` records and rescues file uploads and deletion events

---

## Table of Contents

1. Overview
2. Architecture
3. Quickstart (Django frontend)
4. Running the API (FastAPI)
5. Configuration / .env
6. RAG-specific behaviors
7. Troubleshooting
8. Contributing

---

## 1) Overview

DevTune demonstrates a practical RAG workflow: users upload documents, which are validated and chunked into vectorized embeddings. Chat requests can optionally include those vectors as context to improve responses.

Use cases: research assistants, private document assistants (per-user indexing), prototyping RAG for internal knowledge bases.

---

## 2) Architecture

- Django (`devtune` app): User authentication, UI templates, `KnowledgeFile` model to persist uploaded files and their indexing status.
- FastAPI (`API` folder): Upload, Index, Delete, List and Query endpoints. Contains RAG service implementation using `EmbeddingsService`, `VectorStore` (FAISS/Chroma/In-memory), and `RAGController`.
- Vector storage: FAISS (recommended), with Chromadb or in-memory fallback.
- LLM Providers: `API/stores/llm` contains provider implementations for OpenAI, Cohere, and Ollama. Embedding provider and generation provider are configurable separately (see `.env`).

---

## 3) Quickstart — Django frontend

1. Create and activate a Python environment (conda, venv, or pipx). Example with conda:

```bash
conda create -n devtune python=3.11 -y
conda activate devtune
```

2. Install project dependencies:

```bash
pip install -r requirements.txt
```

3. Apply Django migrations and create a superuser:

```bash
python manage.py migrate
python manage.py createsuperuser
```

4. Start the Django development server:

```bash
python manage.py runserver
```

5. Visit the user interface (default http://127.0.0.1:8000/) and login or sign up to create `KnowledgeFile` records and start uploading documents.

---

## 4) Running the API (FastAPI)

1. Move into the API directory:

```bash
cd API
```

2. Configure required `.env` values (API keys, backends). See section **Configuration** below.
3. Start the FastAPI server (uvicorn):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5555
```

4. Open the interactive docs at http://127.0.0.1:5555/docs to try endpoints.

Common endpoints:

- POST /DevTune/rag/upload — multipart upload with `file`, `username` and `session_id` fields. Uploads are saved under `uploads/<username>/`.
- POST /DevTune/rag/index — indexes a file; you can pass a `file_path` to explicitly index a specific file.
- POST /DevTune/rag/delete — deletes a document by `doc_id` or `file_name` for a given `username` and will compact the FAISS index.
- POST /DevTune/chat/complete — send chat prompts (optionally with `utility_params` including `use_rag` and `rag_top_k`).

---

## 5) Configuration / .env

Set the providers and keys in `API/.env` (or environment variables) for both generation and embeddings.
Recommended variables:

- GENERATION_BACKEND — one of: OPENAI | COHERE | OLLAMA
- EMBEDDING_BACKEND — one of: OPENAI | COHERE | OLLAMA (can be different from GENERATION_BACKEND)
- OPENAI_API_KEY
- OPENAI_MODEL — chat/generation model (e.g., gpt-3.5-turbo)
- OPENAI_EMBEDDING_MODEL — embedding model (e.g., text-embedding-3-small)
- COHERE_API_KEY
- COHERE_MODEL, COHERE_EMBEDDING_MODEL
- OLLAMA_URL, OLLAMA_MODEL, OLLAMA_EMBEDDING_MODEL

Example `.env` snippet:

```
GENERATION_BACKEND=OPENAI
EMBEDDING_BACKEND=OPENAI
OPENAI_API_KEY=sk-yourkey
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 6) RAG-specific behavior — what to expect

- Per-user storage: Files are saved into `uploads/<username>/`. Metadata stored in `KnowledgeFile` associates file_name, saved path and `doc_hash`.
- Auto-index on upload: The API validates and auto-indexes uploads. Indexing creates chunks and embeddings and stores them in the vector DB with `username`/`file_name` metadata.
- Per-user queries: The vector store filters results by `username` metadata to prevent cross-user leakage.
- Deletion and compaction: Deleting a file triggers deletion of its vectors (and associated `doc_id` entries). The vector store runs a `compact_index()` pass to rebuild the FAISS index and purge deleted vectors.
- Provider separation: EmbeddingsService will create and use an embedding-only provider (selected by `EMBEDDING_BACKEND`), while chat generation uses `GENERATION_BACKEND`. This enables using different providers for embeddings and text generation.

---

## 7) Troubleshooting

- If FAISS is not installed, the project will automatically fall back to Chromadb or an in-memory vector store. For production-like performance install FAISS (conda recommended on Windows):

```bash
conda install -c pytorch faiss-cpu -y
```

- Missing environment values: the service will fail silently in some places. Check `API/.env` and server logs for missing keys.
- If you notice `uploads/default` being used, ensure the Django front-end is sending `username` and that upload endpoints validate and persist the username.

---

## 8) Contributing

We welcome contributions! Common ways to contribute:

- Fix a bug or add a test
- Improve the RAG indexing or performance
- Add a new LLM provider

Steps to contribute:

1. Fork the project
2. Create a new branch for your feature/fix
3. Make changes, add tests, and update docs
4. Submit a PR against main with a clear description


�
�"# DevTune_Coding_Assistant"
