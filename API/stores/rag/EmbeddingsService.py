from typing import List
import logging
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.LLMEnums import DocumentTypeEnum


class EmbeddingsService:
    """Service for generating embeddings from text using various providers"""

    def __init__(self, config):
        """
        Initialize EmbeddingsService

        Args:
            config: Configuration object with API keys and model settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.provider = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the embedding provider based on config"""
        try:
            factory = LLMProviderFactory(config=self.config)
            # Prefer an explicit EMBEDDING_BACKEND, otherwise fall back to GENERATION_BACKEND
            embed_backend = getattr(self.config, 'EMBEDDING_BACKEND', None) or getattr(self.config, 'GENERATION_BACKEND', None)
            self.provider = factory.create(provider=embed_backend)
            #self.provider.set_embedding_model(self.config.EMBEDDING_BACKEND)
            # Set embedding model
            if hasattr(self.config, 'OPENAI_EMBEDDING_MODEL') and (getattr(self.config, 'EMBEDDING_BACKEND', '').lower() == "openai"):
                self.provider.set_embedding_model(
                    model_id=self.config.OPENAI_EMBEDDING_MODEL,
                    embedding_size=1536 
                )
            elif hasattr(self.config, 'COHERE_EMBEDDING_MODEL') and (getattr(self.config, 'EMBEDDING_BACKEND', '').lower() == "cohere"):
                self.provider.set_embedding_model(
                    model_id=self.config.COHERE_EMBEDDING_MODEL,
                    embedding_size=1024 
                )
            elif hasattr(self.config, 'OLLAMA_EMBEDDING_MODEL') and (getattr(self.config, 'EMBEDDING_BACKEND', '').lower() == "ollama"):
                self.provider.set_embedding_model(
                    model_id=self.config.OLLAMA_EMBEDDING_MODEL,
                    embedding_size=768  
                )    

            self.logger.info(f"Embeddings provider initialized: {embed_backend}")
        except Exception as e:
            self.logger.error(f"Error initializing embeddings provider: {e}")
            raise

    def embed_text(self, text: str, document_type: str = "document") -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed
            document_type: Type of document ("document" or "query")

        Returns:
            List of floats representing the embedding vector
        """
        if not self.provider:
            raise RuntimeError("Embeddings provider not initialized")

        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")

        try:
            # Convert document_type to appropriate enum
            doc_type = DocumentTypeEnum.QUERY if document_type == "query" else DocumentTypeEnum.DOCUMENT

            embedding = self.provider.embed_text(text, document_type=doc_type)

            if not embedding:
                raise RuntimeError("Failed to generate embedding")

            return embedding

        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise

    def embed_texts(self, texts: List[str], document_type: str = "document") -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed
            document_type: Type of documents ("document" or "query")

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # If underlying provider supports batch embeddings, use it
        try:
            if hasattr(self.provider, 'embed_texts'):
                return self.provider.embed_texts(texts, document_type=document_type)
        except Exception as e:
            self.logger.warning(f"Provider embed_texts failed, falling back to per-text embedding: {e}")

        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_text(text, document_type)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Failed to embed text {i}: {e}")
                # Add None for failed embeddings
                embeddings.append(None)

        return embeddings

    def embed_documents(self, documents: List[dict]) -> List[dict]:
        """
        Generate embeddings for documents with metadata

        Args:
            documents: List of document dicts with 'content' and 'metadata' keys

        Returns:
            List of document dicts with added 'embedding' key
        """
        results = []

        for doc in documents:
            try:
                content = doc.get('content', '')
                if not content:
                    self.logger.warning(f"Empty content in document: {doc.get('metadata', {})}")
                    continue

                embedding = self.embed_text(content, document_type="document")

                result = {
                    'content': content,
                    'metadata': doc.get('metadata', {}),
                    'embedding': embedding
                }
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error embedding document: {e}")
                continue

        return results

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from current provider

        Returns:
            Integer dimension size
        """
        if not self.provider or not hasattr(self.provider, 'embedding_size'):
            # Default dimensions based on embedding backend
            backend = getattr(self.config, 'EMBEDDING_BACKEND', None) or getattr(self.config, 'GENERATION_BACKEND', None)
            if backend and backend.lower() == "openai":
                return 1536
            elif backend and backend.lower() == "cohere":
                return 1024
            else:
                return 768

        return self.provider.embedding_size
