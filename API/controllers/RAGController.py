from typing import List, Dict, Optional
import os
import logging
import uuid
from controllers.BaseController import BaseController
from stores.rag.loaders.DocumentLoader import DocumentLoader
from stores.rag.TextSplitter import TextSplitter
from stores.rag.EmbeddingsService import EmbeddingsService
from stores.rag.VectorStore import VectorStore
from stores.llm.LLMProviderFactory import LLMProviderFactory
from helpers.history import load_history
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class RAGController(BaseController):
    """Controller for RAG (Retrieval-Augmented Generation) operations"""

    def __init__(self, session_id=None, username=None, utility_params=None):
        super().__init__()
        self.session_id = session_id
        self.username = username
        self.utility_params = utility_params or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter(
            chunk_size=self.utility_params.get('chunk_size', 700),
            chunk_overlap=self.utility_params.get('chunk_overlap', 200),
        )
        self.embeddings_service = EmbeddingsService(config=self.app_settings)
        self.vector_store = VectorStore(
            persist_directory="./faiss_db",
            collection_name=self.utility_params.get('collection_name', 'documents')
        )

    async def index_document(self, file_path: str) -> Dict:
        """
        Index a document: load, split, embed, and store

        Args:
            file_path: Path to the document file

        Returns:
            Dict with indexing results
        """
        try:
            # Ensure file exists before indexing (improve error message)
            if not os.path.exists(file_path):
                self.logger.error(f"Indexing failed: file not found: {file_path}")
                return {'success': False, 'message': f'File not found: {file_path}'}
            # Load document
            self.logger.info(f"Loading document: {file_path}")
            document = self.document_loader.load_document(file_path)
            # include username in metadata for per-user isolation
            if document.metadata is None:
                document.metadata = {}
            if self.username:
                document.metadata['username'] = self.username
            doc_hash = document.metadata.get('doc_hash') if document.metadata else None
            if doc_hash and self.vector_store.document_exists_by_hash(doc_hash, username=self.username):
                return {
                    'success': True,
                    'file_path': file_path,
                    'num_chunks': 0,
                    'doc_ids': [],
                    'message': 'Document already indexed (duplicate detected)'
                }
            self.logger.info("Document loaded successfully")
            # Split into chunks
            self.logger.info("Splitting document into chunks")
            chunks = self.text_splitter.split_text(
                text=document.content,
                metadata=document.metadata
            )
            self.logger.info("Document split into chunks")

            # Convert chunks to dict format
            chunk_dicts = []
            for chunk in chunks:
                # Ensure chunk metadata includes doc_hash and a unique chunk_id
                cm = chunk.metadata.copy() if chunk.metadata else {}
                cm['doc_hash'] = doc_hash
                cm['chunk_index'] = cm.get('chunk_index', len(chunk_dicts))
                cm['chunk_id'] = f"{doc_hash}-{cm['chunk_index']}" if doc_hash else cm.get('chunk_id', str(uuid.uuid4()))
                if self.username:
                    cm['username'] = self.username
                # include file_name metadata for filtering
                if document.metadata.get('file_name'):
                    cm['file_name'] = document.metadata.get('file_name')
                chunk_dicts.append({
                    'content': chunk.content,
                    'metadata': cm
                })
            self.logger.info("Chunk dicts created")

            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(chunk_dicts)} chunks")
            embedded_docs = self.embeddings_service.embed_documents(chunk_dicts)

            # Store in vector database
            self.logger.info("Storing chunks in vector database")
            doc_ids = self.vector_store.add_documents(embedded_docs)

            return {
                'success': True,
                'file_path': file_path,
                'num_chunks': len(chunks),
                'doc_ids': doc_ids,
                'message': f"Successfully indexed {len(chunks)} chunks from document"
            }

        except Exception as e:
            self.logger.error(f"Error indexing document: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to index document: {str(e)}"
            }

    async def index_directory(self, directory_path: str) -> Dict:
        """
        Index all documents in a directory

        Args:
            directory_path: Path to directory containing documents

        Returns:
            Dict with indexing results
        """
        try:
            # Load all documents
            self.logger.info(f"Loading documents from directory: {directory_path}")
            documents = self.document_loader.load_directory(directory_path)

            results = []
            total_chunks = 0

            for document in documents:
                # Validate document before processing
                try:
                    temp_file_path = document.metadata.get('source')
                    validation = await self.validate_document(temp_file_path)
                    if not validation.get('success'):
                        self.logger.info(f"Skipping file {document.metadata.get('file_name')} due to validation failure: {validation.get('message')}")
                        results.append({
                            'file_name': document.metadata.get('file_name'),
                            'num_chunks': 0,
                            'doc_ids': [],
                            'skipped_validation': True,
                            'message': validation.get('message')
                        })
                        continue
                except Exception:
                    pass
                # Annotate with username to maintain ownership
                if document.metadata is None:
                    document.metadata = {}
                if self.username:
                    document.metadata['username'] = self.username
                # Ensure file_name is present
                if not document.metadata.get('file_name') and document.metadata.get('source'):
                    try:
                        document.metadata['file_name'] = document.metadata.get('source').split(os.path.sep)[-1]
                    except Exception:
                        pass

                # Skip existing duplicates
                doc_hash = document.metadata.get('doc_hash') if document.metadata else None
                if doc_hash and self.vector_store.document_exists_by_hash(doc_hash, username=self.username):
                    self.logger.info(f"Skipping file {document.metadata.get('file_name')} (duplicate)")
                    results.append({
                        'file_name': document.metadata.get('file_name'),
                        'num_chunks': 0,
                        'doc_ids': [],
                        'skipped_duplicate': True
                    })
                    continue
                # Split into chunks
                chunks = self.text_splitter.split_text(
                    text=document.content,
                    metadata=document.metadata
                )

                # Convert to dict format and add chunk metadata (doc_hash, chunk_id)
                chunk_dicts = []
                for chunk in chunks:
                    cm = chunk.metadata.copy() if chunk.metadata else {}
                    cm['doc_hash'] = doc_hash
                    cm['chunk_index'] = cm.get('chunk_index', len(chunk_dicts))
                    cm['chunk_id'] = f"{doc_hash}-{cm['chunk_index']}" if doc_hash else cm.get('chunk_id', str(uuid.uuid4()))
                    chunk_dicts.append({
                        'content': chunk.content,
                        'metadata': cm
                    })

                # Generate embeddings
                embedded_docs = self.embeddings_service.embed_documents(chunk_dicts)

                # Store in vector database
                doc_ids = self.vector_store.add_documents(embedded_docs)

                results.append({
                    'file_name': document.metadata.get('file_name'),
                    'num_chunks': len(chunks),
                    'doc_ids': doc_ids
                })

                total_chunks += len(chunks)

            return {
                'success': True,
                'directory_path': directory_path,
                'num_documents': len(documents),
                'total_chunks': total_chunks,
                'results': results,
                'message': f"Successfully indexed {len(documents)} documents with {total_chunks} chunks"
            }

        except Exception as e:
            self.logger.error(f"Error indexing directory: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to index directory: {str(e)}"
            }

    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant documents using query

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents
        """
        try:
            # Generate query embedding
            self.logger.info(f"Searching for: {query} (username={self.username}, top_k={top_k})")
            query_embedding = self.embeddings_service.embed_text(query, document_type="query")

            # Search vector store
            filter_metadata = None
            if self.username:
                filter_metadata = {'username': self.username}
            filter_metadata = None
            if self.username:
                filter_metadata = {'username': self.username}
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            self.logger.info(f"Found {len(results)} results for '{query}' (username={self.username})")


            return results

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    async def rag_query(self, query: str, top_k: int = 3) -> str:
        """
        RAG query: retrieve relevant context and generate answer

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated answer
        """
        try:
            # Retrieve relevant documents
            self.logger.info(f"RAG Query: {query} (username={self.username}, top_k={top_k})")
            relevant_docs = await self.search_documents(query, top_k=top_k)

            if not relevant_docs:
                return "I couldn't find any relevant information in the knowledge base to answer your question."
            self.logger.info(f"RAG Query retrieved {len(relevant_docs)} documents; building context")

            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                context_parts.append(f"[Document {i}]\n{doc['content']}\n")

            context = "\n".join(context_parts)

            # Optionally include recent chat history for session context
            try:
                if self.session_id:
                    history = await load_history(self.session_id)
                    if history:
                        histparts = []
                        for h in history[-5:]:
                            histparts.append(f"User: {h[0].content}\nAI: {h[1].content}")
                        context = context + "\n\nRecent Chat History:\n" + "\n".join(histparts)
            except Exception:
                pass

            # Generate answer using LLM
            factory = LLMProviderFactory(config=self.app_settings)
            provider = factory.create(provider=self.app_settings.GENERATION_BACKEND)

            template_str = """
            You are a helpful AI assistant. Use the following context from the knowledge base to answer the user's question.
            If the context doesn't contain relevant information, say so politely.

            Context:
            {context}

            Question: {question}

            Instructions:
            - Provide a clear, accurate answer based on the context
            - Cite specific information from the context when possible
            - If the context doesn't fully answer the question, acknowledge the limitations
            - Keep the answer concise and focused

            Answer:
            """

            template = ChatPromptTemplate.from_template(template_str)
            
            provider_runnable = RunnableLambda(lambda v: provider.generate_text(v.to_string()))
            chain = template | provider_runnable

            response = chain.invoke({
                "context": context,
                "question": query
            })

            return response

        except Exception as e:
            self.logger.error(f"Error in RAG query: {e}")
            return f"An error occurred while processing your query: {str(e)}"

    async def delete_document(self, doc_id: str) -> Dict:
        """
        Delete a document from the vector store

        Args:
            doc_id: Document ID to delete

        Returns:
            Dict with deletion result
        """
        try:
            # Ensure the doc belongs to this username
            doc = self.vector_store.get_document(doc_id)
            if not doc:
                return {'success': False, 'doc_id': doc_id, 'message': 'Document not found'}
            # Allow deletion if the username not set in metadata but file path contains username
            doc_meta = doc.get('metadata', {})
            owner_meta = doc_meta.get('username') or None
            if self.username and owner_meta and owner_meta != self.username:
                return {'success': False, 'doc_id': doc_id, 'message': 'Unauthorized to delete this document'}
            if self.username and not owner_meta:
                # Attempt to deduce owner from the source path if present
                source_path = doc_meta.get('source') or ''
                try:
                    # If the source has /uploads/<username>/, check if it matches
                    parts = source_path.split(os.path.sep)
                    if self.username not in parts:
                        return {'success': False, 'doc_id': doc_id, 'message': 'Unauthorized to delete this document (owner mismatch)'}
                except Exception:
                    return {'success': False, 'doc_id': doc_id, 'message': 'Unauthorized to delete this document (unable to verify owner)'}
            success = self.vector_store.delete_document(doc_id)
            if success:
                self.logger.info(f"Deleted doc {doc_id} from vector store (username={self.username})")
            else:
                self.logger.info(f"Failed to delete doc {doc_id} from vector store (username={self.username})")
            return {
                'success': success,
                'doc_id': doc_id,
                'message': f"Document {doc_id} deleted successfully" if success else "Failed to delete document"
            }
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Error deleting document: {str(e)}"
            }

    async def list_documents(self, limit: int = 100, offset: int = 0) -> Dict:
        """
        List documents in the vector store

        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            Dict with document list
        """
        try:
            # Filter documents by username if provided
            filter_metadata = {'username': self.username} if self.username else None
            documents = self.vector_store.list_documents(limit=limit, offset=offset, filter_metadata=filter_metadata)
            total_count = self.vector_store.count_documents()

            return {
                'success': True,
                'documents': documents,
                'total_count': total_count,
                'limit': limit,
                'offset': offset
            }
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'documents': [],
                'total_count': 0
            }

    async def get_statistics(self) -> Dict:
        """
        Get RAG system statistics

        Returns:
            Dict with system statistics
        """
        try:
            total_docs = self.vector_store.count_documents()

            return {
                'success': True,
                'total_documents': total_docs,
                'collection_name': self.vector_store.collection_name,
                'embedding_dimension': self.embeddings_service.get_embedding_dimension()
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def reset_collection(self) -> Dict:
        """
        Reset vector store collection
        """
        try:
            self.vector_store.reset_collection()
            return {'success': True, 'message': f'Collection {self.vector_store.collection_name} reset'}
        except Exception as e:
            self.logger.error(f"Error resetting collection: {e}")
            return {'success': False, 'error': str(e)}

    async def delete_documents_by_file(self, file_name: str) -> Dict:
        """
        Delete all document chunks for a given file and username
        """
        try:
            if not file_name:
                return {'success': False, 'message': 'file_name required'}
            filter_metadata = {'file_name': file_name}
            if self.username:
                filter_metadata['username'] = self.username
            count = self.vector_store.delete_documents(filter_metadata=filter_metadata)
            # If nothing found, attempt a fallback where metadata doesn't have username but the path indicates the same user
            if count == 0 and self.username:
                # Search vector store docs for matching file_name and try to deduce owner from 'source'
                found_count = 0
                for doc in self.vector_store.documents:
                    md = doc.get('metadata', {})
                    fn = md.get('file_name')
                    source = md.get('source') or ''
                    # Match by recorded file_name OR by the source path ending with the provided file_name
                    if fn == file_name or (source and (source.endswith(file_name) or source.split(os.path.sep)[-1] == file_name)):
                        # If username is present in metadata, ensure it matches (ownership)
                        owner = md.get('username')
                        if owner and self.username and owner != self.username:
                            continue
                        did = doc.get('id')
                        if did:
                            self.vector_store.delete_document(did)
                            found_count += 1
                count = found_count
            return {'success': True, 'deleted_count': count, 'message': f'Deleted {count} documents for file {file_name}'}
        except Exception as e:
            self.logger.error(f"Error deleting documents by file: {e}")
            return {'success': False, 'error': str(e)}

    async def validate_document(self, file_path: str) -> Dict:
        """
        Validate a document: check load, non-empty content, and embedding sample

        Returns a dict with validation status and metadata
        """
        try:
            document = self.document_loader.load_document(file_path)
            if not document or not document.content or len(document.content.strip()) == 0:
                return {'success': False, 'message': 'Document is empty'}

            # Test embedding on a small sample
            sample_text = document.content[:512]
            embedding = None
            try:
                embedding = self.embeddings_service.embed_text(sample_text, document_type='document')
            except Exception as embed_exc:
                return {'success': False, 'message': f'Embedding failed: {embed_exc}'}

            if not embedding or len(embedding) == 0:
                return {'success': False, 'message': 'Embedding returned empty vector'}

            # Dimension check vs vector store
            embed_dim = len(embedding)
            vec_dim = self.vector_store.dimension
            if vec_dim is not None and embed_dim != vec_dim:
                return {'success': False, 'message': f'Embedding dimension mismatch (vector store: {vec_dim}, embedding: {embed_dim})'}

            # Duplicate check
            doc_hash = document.metadata.get('doc_hash') if document.metadata else None
            if doc_hash and self.vector_store.document_exists_by_hash(doc_hash, username=self.username):
                return {'success': True, 'skipped_duplicate': True, 'message': 'Document is duplicate and already indexed'}

            # Bill of materials: content size and length checks
            max_chars = getattr(self.app_settings, 'INPUT_DEFAULT_MAX_CHARACTERS', 2000)
            if len(document.content) > (max_chars * 50):
                return {'success': True, 'large': True, 'message': f'Document is very large ({len(document.content)} chars); indexing will be chunked.'}

            return {'success': True, 'message': 'Document validated', 'doc_hash': doc_hash}

        except Exception as e:
            self.logger.error(f"Error validating document: {e}")
            return {'success': False, 'message': str(e)}
