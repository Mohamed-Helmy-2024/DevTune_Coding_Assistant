++ Begin Patch
# DevTune API — FastAPI RAG Service

This folder contains the FastAPI service and the RAG (Retrieval-Augmented Generation) implementation used by the DevTune Django frontend.

This service provides endpoints for:
- Uploading new documents
- Validating and indexing documents into a vector database (FAISS, Chromadb, or in-memory fallback)
- Performing vectorized searches to provide RAG context for chat completions
- Deleting documents and compaction of the vector index

---

## Table of Contents
1. Features
2. Quickstart
3. Environment Variables and Configuration
4. Endpoints & Example Requests
5. How RAG Works
6. Admin / Maintenance
7. Troubleshooting
8. Contributing

---

## 1) Features
- Per-user uploads and indexing (uploads are saved to `uploads/<username>`)
- Auto-indexing of uploads to keep the vector store in sync
- Optional RAG context in chat queries via a `use_rag` toggle or `utility_params`
- Configurable embedding vs generation backends so you can use different providers
- FAISS index compaction after deletes to remove deleted vectors

---

## 2) Quickstart
```bash
# Create environment and install requirements
conda create -n devtune_api python=3.11 -y
conda activate devtune_api
pip install -r requirements.txt

# Configure your .env file with provider keys (see Configuration section)
# Run the API server
uvicorn main:app --reload --host 0.0.0.0 --port 5555
```

Open Swagger UI at: http://127.0.0.1:5555/docs

---

## 3) Environment Variables and Configuration
Set the following environment variables in `API/.env` or the environment used to run the API service:
- GENERATION_BACKEND (OPENAI | COHERE | OLLAMA)
- EMBEDDING_BACKEND (OPENAI | COHERE | OLLAMA) — optional, defaults to GENERATION_BACKEND if unset
- OPENAI_API_KEY and OPENAI_MODEL and OPENAI_EMBEDDING_MODEL
- COHERE_API_KEY and COHERE_MODEL and COHERE_EMBEDDING_MODEL
- OLLAMA_URL and OLLAMA_MODEL and OLLAMA_EMBEDDING_MODEL

The app uses a `Settings` model (`API/helpers/configs.py`) that loads these values at startup. The `LLMProviderFactory` builds providers for the selected backend(s).

---

## 4) Endpoints & Example Requests
Use the OpenAPI UI for a complete list but here are common ones:

- Upload a file
  - POST /DevTune/rag/upload
  - Request type: multipart/form-data
  - Fields: `file` (binary), `username` (string), `session_id` (string)
  - Returns: `file_path`, `saved_file_name`, indexing status — may auto-index the file
  - Example curl:
  ```bash
  curl -F "file=@example.pdf" -F "username=alice" -F "session_id=abc123" http://127.0.0.1:5555/DevTune/rag/upload
  ```

- Index an existing file
  - POST /DevTune/rag/index
  - Request type: JSON
  - Body: {"file_path": "uploads/alice/example.pdf", "username": "alice"}

- Search / Query
  - POST /DevTune/rag/query
  - Body: {"query": "How do I use sockets in Python?", "top_k": 3, "username": "alice"}

- Chat completion (with optional RAG)
  - POST /DevTune/chat/complete
  - Body: {
    "username": "alice",
    "session_id": "abc123",
    "prompt": "Explain recursion",
    "utility_params": {"completion_type": "main", "use_rag": true, "rag_top_k": 3}
  }

- Delete documents by file or doc id
  - POST /DevTune/rag/delete
  - Body options: {"doc_id": "...", "username": "alice"} OR {"file_name": "example.pdf", "username": "alice"}
  - Deletes vectors associated with the given file/doc and then compacts the index to remove vector entries.

---

## 5) How RAG Works
1. Upload/Index: Upload a file via the API or via the Django frontend. The `RAGController` validates file ownership, reads the file, computes SHA256 (doc_hash), splits the document into chunks, creates embeddings via the EmbeddingsService and stores them in the VectorStore.
2. Metadata: Each chunk is stored along with metadata (username, file_name, doc_hash, chunk_index). Searches can filter on these fields to provide per-user isolation.
3. Query & Chat: ChatController can include RAG results in the prompt when `utility_params.use_rag` is true — this integrates search hits into the prompt provided to the generation provider.
4. Deletion & compaction: When a doc or file is deleted, the API marks entries as deleted, then runs an index compaction to rebuild the FAISS index with only active vectors. This reduces index size and ensures removed content is no longer searchable.

---

## 6) Admin / Maintenance
- Backup: Persisted data in `faiss_db` contains saved index and metadata; ensure this is backed up if you rely on the prebuilt index.
- Rebuild index: If the index becomes inconsistent or corrupted, use an `index_directory` call or script to reindex all `uploads/<username>` files.
- Migrate default uploads: If you have legacy files in `uploads/default`, move them into each user's `uploads/<username>` and reindex.

---

## 7) Troubleshooting
- Missing embeddings: If embedding calls fail confirm the provider's API key and embedding model are configured. Check server logs for `EmbeddingsService` errors.
- Duplicate files: Uploads are filtered by the doc_hash (SHA256). If you see no new indexing, ensure the file content changed, or force reindexing by adding a unique suffix.
- Index compaction and performance: Rebuilding FAISS indexes can be CPU intensive; for production, consider running compaction as a background job or leveraging FAISS's id remapping to avoid frequent heavy compaction.

---

## 8) Contributing
- Add new providers, upgrade models, or enhance indexing (e.g. more loaders or better chunkers).
- When making changes to RAG logic, add integration tests that upload, index and query documents, verifying per-user isolation and deletion.

---

If you need hands-on support or a deeper explanation of the RAG pipeline, adding small example scripts for quick testing is recommended.

License: MIT

++ End Patch
1. `conda create -n  DEVTUNE python=3.11`
2. `conda activate DEVTUNE`

```
conda create --name new_environment_name --clone existing_environment_name
conda remove --name <environment_name> --all
```
 
## Quick RAG & Chat Usage

1. Set your generation backend and API keys in `.env`. Example:

```
GENERATION_BACKEND=OPENAI
OPENAI_API_KEY=your_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-3.5-turbo
```

2. Start the server:
```
uvicorn main:app --reload --host 0.0.0.0 --port 5555
```

3. RAG endpoints:
- Upload: POST /DevTune/rag/upload (multipart/form-data file, username, session_id)
- Index: POST /DevTune/rag/index (JSON body with file_name, username, session_id)
- Query: POST /DevTune/rag/query (query body)

4. Chat endpoints:
- Chat: POST /DevTune/chat/complete with JSON body including prompt, session_id and optional utility_params. Example:
```
{
	"username": "test",
	"session_id": "session123",
	"prompt": "Explain recursion",
	"utility_params": {"completion_type": "main", "use_rag": true, "rag_top_k": 3}
}
```

Notes:
- Uploaded files are validated with a quick embedding test and duplicates are skipped based on a SHA256 hash stored in metadata.
- The system uses LangChain loaders and text splitters when available to create robust chunks for indexing.
- Chat history is persisted in the DB; RAG queries can be augmented with recent conversation history for personalized responses.

Installing FAISS (optional but recommended):

- Linux (pip):
```
pip install faiss-cpu
```
- Windows (recommended via conda):
```
conda install -c pytorch faiss-cpu
```
- If FAISS is not available, the system automatically falls back to Chromadb (if installed) or an in-memory vector store for development and testing.
requirements.txt

3. `pip install -r requirements.txt`

4. `uvicorn main:app --reload --host 0.0.0.0 --port 5555`



5.contrib

```
# Assuming you're starting from scratch
git clone https://github.com/Ali-Abdelhamid-Ali/DEPI-PROJECT.git
cd DEPI-PROJECT

# Create your feature branch
git checkout -b feature-chat-history

# Copy your code files to this directory
# Then add and commit
git add .
git commit -m "Implement chat history with conversation memory"

# Push to the specific branch
git push -u origin feature-chat-history
```