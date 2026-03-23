from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging
import uuid
import os
import shutil
from pathlib import Path

from .schemas.rag import (
    RAGQueryRequest,
    DocumentIndexRequest,
    DocumentSearchRequest,
    DocumentDeleteRequest,
    DocumentListRequest,
    DocumentUploadResponse
)
from helpers.configs import Settings, get_settings
from controllers.RAGController import RAGController

logger = logging.getLogger('uvicorn.error')

rag_router = APIRouter(
    prefix=f"/{get_settings().APP_NAME}/rag",
    tags=["RAG"]
)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@rag_router.post("/query")
async def rag_query(request: RAGQueryRequest, app_settings: Settings = Depends(get_settings)):
    """
    RAG Query endpoint: retrieve relevant context and generate answer
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username,
            utility_params=request.utility_params
        )
            
        response = await controller.rag_query(
            query=request.query,
            top_k=request.top_k
        )

        return JSONResponse({
            "session_id": request.session_id,
            "username": request.username,
            "query": request.query,
            "response": response
        })

    except Exception as e:
        logger.exception("RAG query error:")
        raise HTTPException(status_code=500, detail=str(e))
    
@rag_router.post("/search")
async def search_documents(request: DocumentSearchRequest, app_settings: Settings = Depends(get_settings)):
    """
    Search for relevant documents without generating answer
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username
        )

        results = await controller.search_documents(
            query=request.query,
            top_k=request.top_k
        )

        return JSONResponse({
            "session_id": request.session_id,
            "username": request.username,
            "query": request.query,
            "results": results,
            "num_results": len(results)
        })

    except Exception as e:
        logger.exception("Search documents error:")
        raise HTTPException(status_code=500, detail=str(e))



@rag_router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    username: str = Form(None)
):
    """
    Upload a document file
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        allowed_extensions = ['.txt', '.pdf', '.docx', '.md']

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Require username to ensure per-user upload directories
        if not username or not username.strip():
            logger.error("Upload called without username - refusing to save to default")
            raise HTTPException(status_code=400, detail="username is required")

        # Create sanitized user-specific directory
        uname = username.strip()
        uname = os.path.basename(uname) or "default"
        user_dir = os.path.join(UPLOAD_DIR, uname)
        os.makedirs(user_dir, exist_ok=True)

        # Sanitize filename and save file
        filename = os.path.basename(file.filename) or (str(uuid.uuid4()) + ".upload")
        file_path = os.path.join(user_dir, filename)

        # Basic file size check (20MB). Prevent very large uploads by default.
        file_size_limit = 20 * 1024 * 1024  # 20 MB
        contents = await file.read()
        if len(contents) > file_size_limit:
            raise HTTPException(status_code=400, detail=f"File is too large. Maximum allowed size is {file_size_limit} bytes")

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        # Reset the file pointer to beginning if further use is needed
        try:
            file.file.seek(0)
        except Exception:
            pass

        # Validate uploaded document (quick checks + embedding test) using RAGController
        controller = RAGController(session_id=session_id, username=username)
        validation = await controller.validate_document(file_path)

        # Attempt to auto-index the file after successful validation
        index_result = None
        if validation.get('success') and not validation.get('skipped_duplicate'):
            try:
                # If controller.index_document expects a path, ensure username is passed
                index_result = await controller.index_document(file_path)
            except Exception:
                logger.exception('Auto-index failed')

        logger.info(f"Saved file to: {file_path} (username={username}, session={session_id}, size={len(contents)})")

        return JSONResponse({
            "success": True,
            "file_name": file.filename,
            "saved_file_name": filename,
            "file_path": file_path,
            "message": f"File {file.filename} uploaded successfully",
            "validation": validation,
            "index_result": index_result
        })

    except Exception as e:
        logger.exception("Upload document error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/index")
async def index_document(request: DocumentIndexRequest, app_settings: Settings = Depends(get_settings)):
    """
    Index an uploaded document
    """
    try:
        # Construct file path using explicit `file_path` if provided, else derive from username & file_name
        if request.file_path:
            file_path = request.file_path
        else:
            user_dir = os.path.join(UPLOAD_DIR, request.username or "default")
            file_path = os.path.join(user_dir, request.file_name or "")
        logger.info(f"Index request for file path: {file_path} (username={request.username}, file_name={request.file_name})")

        if not os.path.exists(file_path):
            # Fallback to search in user-specific upload directory by file_name
            if request.file_name:
                fallback_path = os.path.join(UPLOAD_DIR, (request.username or 'default'), request.file_name)
                if os.path.exists(fallback_path):
                    file_path = fallback_path
                else:
                    # Search across all user upload subdirectories for the file
                    found = None
                    for root, dirs, files in os.walk(UPLOAD_DIR):
                        if request.file_name in files:
                            found = os.path.join(root, request.file_name)
                            break
                    if found:
                        file_path = found
            if not os.path.exists(file_path):
                logger.error(f"Index called but file not found: {file_path} (username={request.username}, file_name={request.file_name})")
                raise HTTPException(status_code=404, detail=f"File not found: {request.file_name}")

        controller = RAGController(
            session_id=request.session_id,
            username=request.username,
            utility_params=request.utility_params
        )
        # Validate document before indexing
        validation = await controller.validate_document(file_path)
        if not validation.get('success'):
            raise HTTPException(status_code=400, detail=f"Validation failed: {validation.get('message')}")

        # Ownership check: The file must reside inside the user's upload directory
        try:
            user_dir = os.path.abspath(os.path.join(UPLOAD_DIR, request.username or "default"))
            file_abs = os.path.abspath(file_path)
            if os.path.commonpath([file_abs, user_dir]) != user_dir:
                logger.error(f"Index request for file outside user directory: {file_abs} (user_dir={user_dir})")
                raise HTTPException(status_code=403, detail=f"Not authorized to index this file: {request.file_name}")
        except ValueError:
            # On some systems, commonpath may raise ValueError if paths are on different drives; treat as unauthorized
            logger.error(f"Index ownership check failed; file not inside user dir: {file_path}")
            raise HTTPException(status_code=403, detail=f"Not authorized to index this file: {request.file_name}")

        result = await controller.index_document(file_path)

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Index document error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/index-directory")
async def index_directory(request: DocumentIndexRequest, app_settings: Settings = Depends(get_settings)):
    """
    Index all documents in user's upload directory
    """
    try:
        user_dir = os.path.join(UPLOAD_DIR, request.username or "default")

        if not os.path.exists(user_dir):
            raise HTTPException(status_code=404, detail=f"Directory not found for user: {request.username}")

        controller = RAGController(
            session_id=request.session_id,
            username=request.username,
            utility_params=request.utility_params
        )

        result = await controller.index_directory(user_dir)

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Index directory error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/delete")
async def delete_document(request: DocumentDeleteRequest, app_settings: Settings = Depends(get_settings)):
    """
    Delete a document from the vector store
    """
    try:
        logger.info(f"API delete called with doc_id={request.doc_id}, file_name={request.file_name}, username={request.username}")
        controller = RAGController(
            session_id=request.session_id,
            username=request.username
        )

        # Support deleting by doc_id or by file_name (all chunks for that file)
        if request.doc_id:
            result = await controller.delete_document(request.doc_id)
        elif request.file_name:
            result = await controller.delete_documents_by_file(request.file_name)
        else:
            raise HTTPException(status_code=400, detail='doc_id or file_name is required')

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Delete document error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/list")
async def list_documents(request: DocumentListRequest, app_settings: Settings = Depends(get_settings)):
    """
    List documents in the vector store
    """
    try:
        controller = RAGController(
            session_id=request.session_id,
            username=request.username
        )

        result = await controller.list_documents(
            limit=request.limit,
            offset=request.offset
        )

        return JSONResponse(result)

    except Exception as e:
        logger.exception("List documents error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.get("/stats")
async def get_statistics(app_settings: Settings = Depends(get_settings)):
    """
    Get RAG system statistics
    """
    try:
        controller = RAGController()
        result = await controller.get_statistics()

        return JSONResponse(result)

    except Exception as e:
        logger.exception("Get statistics error:")
        raise HTTPException(status_code=500, detail=str(e))


@rag_router.post("/reset")
async def reset_collection(app_settings: Settings = Depends(get_settings)):
    try:
        controller = RAGController()
        result = await controller.reset_collection()
        return JSONResponse(result)
    except Exception as e:
        logger.exception("Reset collection error:")
        raise HTTPException(status_code=500, detail=str(e))
