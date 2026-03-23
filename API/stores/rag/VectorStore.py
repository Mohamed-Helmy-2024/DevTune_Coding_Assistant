import os
import json
import logging
import pickle
from typing import List, Dict, Optional
import numpy as np
import uuid
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    chromadb = None
    CHROMA_AVAILABLE = False


class VectorStore:
    """Vector store for storing and retrieving document embeddings using FAISS"""

    def __init__(self, persist_directory: str = "./faiss_db", collection_name: str = "documents"):
        """
        Initialize VectorStore

        Args:
            persist_directory: Directory to persist the vector database
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # File paths
        self.index_path = os.path.join(persist_directory, f"{collection_name}.index")
        self.data_path = os.path.join(persist_directory, f"{collection_name}_data.pkl")

        # Initialize storage
        self.index = None
        self.documents = []  # List of {id, content, metadata}
        self.id_to_idx = {}  # Map document ID to index position
        self.dimension = None
        # Backend choice: faiss, chroma, or in-memory
        if FAISS_AVAILABLE:
            self.backend = 'faiss'
        elif CHROMA_AVAILABLE:
            self.backend = 'chroma'
        else:
            self.backend = 'in_memory'

        self._chroma_client = None
        self._chroma_collection = None

        # Load existing data if available
        self._load()
        if self.backend == 'in_memory' and not hasattr(self, 'embeddings'):
            self.embeddings = []

        self.logger.info(f"VectorStore initialized with collection: {collection_name}")

    def _load(self):
        """Load existing index and data from disk"""
        try:
            if self.backend == 'faiss' and os.path.exists(self.index_path) and os.path.exists(self.data_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    self.index = None
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.dimension = data.get('dimension')
                    # load embeddings when present for FAISS to allow rebuilds
                    self.embeddings = data.get('embeddings', [])
            elif self.backend == 'chroma':
                try:
                    # initialize chroma client and collection
                    client = chromadb.Client()
                    self._chroma_client = client
                    # try to get collection if exists
                    self._chroma_collection = client.get_collection(self.collection_name)
                except Exception:
                    # create if not exist
                    try:
                        self._chroma_collection = client.create_collection(self.collection_name)
                        self._chroma_client = client
                    except Exception:
                        self._chroma_collection = None
            else:
                # in-memory: load from pickle if exists
                if os.path.exists(self.data_path):
                    with open(self.data_path, 'rb') as f:
                        data = pickle.load(f)
                        self.documents = data.get('documents', [])
                        self.id_to_idx = data.get('id_to_idx', {})
                        self.dimension = data.get('dimension')
                        self.embeddings = data.get('embeddings', [])
                self.logger.info(f"Loaded {len(self.documents)} documents from disk")
        except Exception as e:
            self.logger.warning(f"Could not load existing data: {e}")

    def _save(self):
        """Save index and data to disk"""
        try:
            if self.backend == 'faiss' and self.index is not None:
                faiss.write_index(self.index, self.index_path)
                with open(self.data_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'id_to_idx': self.id_to_idx,
                        'dimension': self.dimension,
                        'embeddings': getattr(self, 'embeddings', [])
                    }, f)
            elif self.backend == 'in_memory':
                with open(self.data_path, 'wb') as f:
                    pickle.dump({
                        'documents': self.documents,
                        'id_to_idx': self.id_to_idx,
                        'dimension': self.dimension,
                        'embeddings': getattr(self, 'embeddings', [])
                    }, f)
                self.logger.info(f"Saved {len(self.documents)} documents to disk")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")

    def _initialize_index(self, dimension: int):
        """Initialize FAISS index with given dimension"""
        self.dimension = dimension
        # Using IndexFlatIP for cosine similarity (normalize vectors first)
        if self.backend == 'faiss':
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None
        self.logger.info(f"Initialized FAISS index with dimension {dimension}")

    def add_documents(self, documents: List[Dict]) -> List[str]:
        """
        Add documents to the vector store

        Args:
            documents: List of dicts with 'content', 'embedding', and 'metadata' keys

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        try:
            ids = []
            embeddings = []
            docs_to_add = []

            for doc in documents:
                if not doc.get('embedding'):
                    self.logger.warning(f"Document missing embedding: {doc.get('metadata', {})}")
                    continue

                # Duplicate detection via doc_hash in metadata (per-user)
                doc_hash = doc.get('metadata', {}).get('doc_hash')
                meta_username = doc.get('metadata', {}).get('username')
                if doc_hash and self.document_exists_by_hash(doc_hash, username=meta_username):
                    # Skip duplicates
                    self.logger.info(f"Skipping duplicate document with hash: {doc_hash}")
                    continue

                # Generate unique ID
                doc_id = doc.get('metadata', {}).get('id', str(uuid.uuid4()))
                ids.append(doc_id)

                # Extract embedding
                embedding = np.array(doc['embedding'], dtype=np.float32)
                
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)

                # Store document data
                docs_to_add.append({
                    'id': doc_id,
                    'content': doc['content'],
                    'metadata': self._clean_metadata(doc.get('metadata', {}))
                })

            if not ids:
                self.logger.warning("No valid documents to add")
                return []

            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            # Ensure normalized python lists for storage
            normalized = [(e / (np.linalg.norm(e) + 1e-12)).tolist() for e in embeddings_array]

            # Initialize index or set dimension if needed
            if self.backend == 'faiss' and self.index is None:
                self._initialize_index(embeddings_array.shape[1])
            if self.backend == 'chroma' and self._chroma_collection is None and self.dimension is None:
                self.dimension = embeddings_array.shape[1]
            if self.backend == 'in_memory' and self.dimension is None:
                self.dimension = embeddings_array.shape[1]
                self.embeddings = []
            # Validate dimensions
            if embeddings_array.shape[1] != self.dimension:
                # If index is empty, we can reinitialize it to match new dimension
                if self.index is None or len(self.documents) == 0:
                    self.logger.info(f"Reinitializing FAISS index to new embedding dimension {embeddings_array.shape[1]}")
                    self._initialize_index(embeddings_array.shape[1])
                else:
                    self.logger.error(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings_array.shape[1]}")
                    raise ValueError("Embedding dimension mismatch between provider and vector store")

            if self.backend == 'faiss':
                self.index.add(embeddings_array)
                # Append normalized embeddings into self.embeddings for rebuilds
                self.embeddings = getattr(self, 'embeddings', [])
                self.embeddings.extend(normalized)
            elif self.backend == 'chroma' and self._chroma_collection is not None:
                # prepare metadatas and ids
                metadatas = [d.get('metadata', {}) for d in docs_to_add]
                contents = [d.get('content') for d in docs_to_add]
                ids_to_upsert = ids
                try:
                    self._chroma_collection.add(ids=ids_to_upsert, metadatas=metadatas, documents=contents, embeddings=embeddings_array.tolist() if hasattr(embeddings_array,'tolist') else None)
                except Exception as e:
                    self.logger.warning(f"Chroma add failed: {e}")
            else:
                # in_memory: extend arrays and normalize
                self.embeddings = getattr(self, 'embeddings', [])
                self.embeddings.extend(normalized)

            # Store document data
            for i, doc in enumerate(docs_to_add):
                idx = len(self.documents)
                self.documents.append(doc)
                self.id_to_idx[doc['id']] = idx
                # if we have normalized embeddings, ensure self.embeddings has same aligned ordering
                if hasattr(self, 'embeddings') and len(self.embeddings) < len(self.documents):
                    # If backend was faiss and self.embeddings wasn't populated earlier, append from embeddings list
                    try:
                        emb = normalized[i]
                        self.embeddings.append(emb)
                    except Exception:
                        # fallback: append zeros if something is wrong
                        self.embeddings.append(np.zeros(self.dimension, dtype=np.float32).tolist() if self.dimension else [])

            # Save to disk
            self._save()

            self.logger.info(f"Added {len(ids)} documents to vector store")
            return ids

        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise

    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata to ensure compatibility

        Args:
            metadata: Raw metadata dict

        Returns:
            Cleaned metadata dict
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif value is None:
                cleaned[key] = ""
            else:
                cleaned[key] = str(value)
        return cleaned

    def document_exists_by_hash(self, doc_hash: str, username: str = None) -> bool:
        """
        Check if a document with specified doc_hash already exists

        Args:
            doc_hash: SHA256 hash string

        Returns:
            True if exists
        """
        for doc in self.documents:
            if doc['metadata'].get('doc_hash') == doc_hash:
                if username:
                    if doc['metadata'].get('username') == username:
                        return True
                else:
                    return True
        return False

    def search(self, query_embedding: List[float], top_k: int = 5,
               filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for similar documents using query embedding

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (basic support)

        Returns:
            List of matching documents with scores
        """
        try:
            if self.backend == 'faiss' and (self.index is None or self.index.ntotal == 0):
                self.logger.warning("Index is empty")
                return []

            # Prepare query
            query = np.array([query_embedding], dtype=np.float32)
            # Normalize for cosine similarity
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            scores = []
            indices = []
            if self.backend == 'faiss':
                scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
            elif self.backend == 'chroma' and self._chroma_collection is not None:
                # chroma search API - use query
                try:
                    results = self._chroma_collection.query(query_embeddings=query.tolist(), n_results=top_k, include=['metadatas', 'documents', 'distances'])
                    # results structure: documents, metadatas, distances
                    formatted = []
                    docs = results.get('documents', [[]])[0]
                    dists = results.get('distances', [[]])[0]
                    mds = results.get('metadatas', [[]])[0]
                    for i, doc_content in enumerate(docs):
                        formatted.append({'content': doc_content, 'metadata': mds[i], 'score': 1 - float(dists[i])})
                    return formatted
                except Exception as e:
                    self.logger.warning(f"Chroma query failed: {e}")
                    return []
            else:
                # in_memory brute force cosine similarity
                if not hasattr(self, 'embeddings') or len(self.embeddings) == 0:
                    return []
                # compute dot products
                q = query[0]
                dists = np.dot(np.vstack(self.embeddings), q)
                # get top K indices
                top_idx = np.argsort(-dists)[:top_k]
                scores = [[float(dists[i]) for i in top_idx]]
                indices = [[int(i) for i in top_idx]]

            # Format results
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < 0 or idx >= len(self.documents):
                    continue

                doc = self.documents[idx]
                # Skip if marked deleted
                if doc['metadata'].get('deleted'):
                    continue

                # Apply metadata filter if provided
                if filter_metadata:
                    match = all(
                        doc['metadata'].get(k) == v 
                        for k, v in filter_metadata.items()
                    )
                    if not match:
                        continue

                result = {
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(score)
                }
                formatted_results.append(result)

            self.logger.info(f"Found {len(formatted_results)} matching documents for filter {filter_metadata}")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching vector store: {e}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID (marks as deleted, requires rebuild for actual removal)

        Args:
            doc_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                # Preserve existing metadata where possible and mark as deleted
                existing = self.documents[idx] if idx < len(self.documents) else {'id': doc_id, 'content': '', 'metadata': {}}
                meta = existing.get('metadata', {}) or {}
                meta['deleted'] = True
                # Clear content but keep metadata (username, file_name, etc.) so ownership/filtering still works
                self.documents[idx] = {
                    'id': doc_id,
                    'content': '',
                    'metadata': meta
                }
                # Remove id mapping so this id is no longer considered active
                try:
                    del self.id_to_idx[doc_id]
                except Exception:
                    pass
                # For chroma, remove from collection by id
                if self.backend == 'chroma' and self._chroma_collection is not None:
                    try:
                        self._chroma_collection.delete(ids=[doc_id])
                    except Exception:
                        pass
                self._save()
                self.logger.info(f"Marked document as deleted: {doc_id}")
                try:
                    # Rebuild index to remove physically from FAISS
                    self.compact_index()
                except Exception as e:
                    self.logger.warning(f"compact_index failed after delete: {e}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def delete_documents(self, filter_metadata: Dict = None) -> int:
        """
        Delete documents matching filter criteria

        Args:
            filter_metadata: Metadata filters

        Returns:
            Number of documents deleted
        """
        try:
            count = 0
            if filter_metadata:
                for doc_id, idx in list(self.id_to_idx.items()):
                    # guard index bounds
                    if idx < 0 or idx >= len(self.documents):
                        continue
                    doc = self.documents[idx]
                    md = doc.get('metadata', {}) or {}
                    # Use string comparison for robustness
                    match = all(
                        str(md.get(k)) == str(v)
                        for k, v in (filter_metadata or {}).items()
                    )
                    if match:
                        self.logger.info(f"Deleting doc {doc_id} matching metadata filter {filter_metadata}")
                        self.delete_document(doc_id)
                        count += 1
            self.logger.info(f"Deleted {count} documents")
            # After marking deleted documents, compact/rebuild the FAISS index to physically remove them
            try:
                self.compact_index()
            except Exception as e:
                self.logger.warning(f"compact_index failed: {e}")
            return count
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return 0

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Get a document by ID

        Args:
            doc_id: Document ID

        Returns:
            Document dict or None
        """
        try:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                doc = self.documents[idx]
                if not doc['metadata'].get('deleted'):
                    return {
                        'id': doc['id'],
                        'content': doc['content'],
                        'metadata': doc['metadata']
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error getting document {doc_id}: {e}")
            return None

    def list_documents(self, limit: int = 100, offset: int = 0, filter_metadata: Dict = None) -> List[Dict]:
        """
        List all documents in the collection

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            List of documents
        """
        try:
            # Filter out deleted documents and apply metadata filters if provided
            active_docs = [
                doc for doc in self.documents 
                if not doc['metadata'].get('deleted')
            ]
            if filter_metadata:
                filtered = []
                for doc in active_docs:
                    match = True
                    for k, v in filter_metadata.items():
                        if str(doc['metadata'].get(k)) != str(v):
                            match = False
                            break
                    if match:
                        filtered.append(doc)
                active_docs = filtered
            
            # Apply pagination
            start = offset
            end = min(offset + limit, len(active_docs))
            
            return [
                {
                    'id': doc['id'],
                    'content': doc['content'],
                    'metadata': doc['metadata']
                }
                for doc in active_docs[start:end]
            ]
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []

    def count_documents(self) -> int:
        """
        Get total number of documents in the collection

        Returns:
            Document count
        """
        try:
            return len([
                doc for doc in self.documents 
                if not doc['metadata'].get('deleted')
            ])
        except Exception as e:
            self.logger.error(f"Error counting documents: {e}")
            return 0

    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        try:
            self.index = None
            self.documents = []
            self.id_to_idx = {}
            self.dimension = None
            
            # Delete files
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.data_path):
                os.remove(self.data_path)
            if self.backend == 'chroma' and self._chroma_client is not None:
                try:
                    self._chroma_client.delete_collection(self.collection_name)
                except Exception:
                    try:
                        # fallback to deleting contents
                        collection = self._chroma_client.get_collection(self.collection_name)
                        collection.delete()
                    except Exception:
                        pass
                
            self.logger.info(f"Collection {self.collection_name} reset")
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            raise

    def compact_index(self):
        """
        Rebuild the vector index from active (non-deleted) documents and embeddings.
        This physically removes vectors from FAISS index when documents are deleted.
        """
        try:
            active_docs = []
            active_embeddings = []
            for i, doc in enumerate(self.documents):
                if not doc['metadata'].get('deleted'):
                    active_docs.append(doc)
                    # Try to get embedding from self.embeddings
                    emb = None
                    if hasattr(self, 'embeddings') and i < len(self.embeddings):
                        emb = self.embeddings[i]
                    if emb is not None and len(emb) > 0:
                        active_embeddings.append(np.array(emb, dtype=np.float32))
            if not active_docs:
                # Nothing left, reset collection
                self.reset_collection()
                return
            # Rebuild in-memory lists
            self.documents = active_docs
            # Re-create id_to_idx
            self.id_to_idx = {doc['id']: idx for idx, doc in enumerate(self.documents)}
            # Re-create embeddings storage
            self.embeddings = [e.tolist() if isinstance(e, np.ndarray) else list(e) for e in active_embeddings]
            # Rebuild FAISS index if applicable
            if self.backend == 'faiss':
                dim = active_embeddings[0].shape[0]
                self._initialize_index(dim)
                emb_arr = np.vstack(active_embeddings).astype(np.float32)
                # ensure normalized
                norms = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-12
                emb_arr = emb_arr / norms
                self.index.add(emb_arr)
            elif self.backend == 'in_memory':
                # ensure embeddings are normalized
                self.embeddings = [np.array(e, dtype=np.float32) / (np.linalg.norm(e) + 1e-12) for e in self.embeddings]
                self.embeddings = [e.tolist() for e in self.embeddings]
            # Save state
            self._save()
            self.logger.info(f"Compacted index: {len(self.documents)} active documents remain")
        except Exception as e:
            self.logger.error(f"Failed to compact index: {e}")
            raise