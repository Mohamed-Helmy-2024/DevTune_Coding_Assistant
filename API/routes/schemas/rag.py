from pydantic import BaseModel
from typing import Optional, Dict


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query"""
    session_id: str
    username: str
    query: str
    utility_params: Optional[Dict] = {}
    top_k: Optional[int] = 5


class DocumentIndexRequest(BaseModel):
    """Request schema for indexing a document"""
    session_id: str
    username: str
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    utility_params: Optional[Dict] = {}

class DocumentSearchRequest(BaseModel):
    """Request schema for searching documents"""
    session_id: str
    username: str
    query: str
    top_k: Optional[int] = 5


class DocumentDeleteRequest(BaseModel):
    """Request schema for deleting a document by doc id or file name"""
    session_id: str
    username: str
    doc_id: Optional[str] = None
    file_name: Optional[str] = None


class DocumentListRequest(BaseModel):
    """Request schema for listing documents"""
    session_id: str
    username: str
    limit: Optional[int] = 100
    offset: Optional[int] = 0


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload"""
    success: bool
    file_name: str
    file_path: str
    message: str
