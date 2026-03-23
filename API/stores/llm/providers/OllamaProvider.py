from fastapi import requests
from ..LLMInterface import LLMInterface
import ollama
import logging
import requests

class OllamaProvider(LLMInterface):

    def __init__(self,
                 host: str = "http://localhost:11434",
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):

        self.host = host
        self.client = ollama.Client(host=host)
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        # Models
        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.logger = logging.getLogger(__name__)
        self.logger.info("OllamaProvider initialized with host: %s", host)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int, OLLAMA_EMBEDDING_MODEL: str = "embed-english-v3.0"):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.OLLAMA_EMBEDDING_MODEL=OLLAMA_EMBEDDING_MODEL

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def embed_text(self, text: str, document_type: str = None):  # ✅ أضف هذه الدالة
        """Generate embeddings using Ollama"""
        if not self.embedding_model_id:
            self.logger.error("Embedding model not set")
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model_id,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                return embedding
            else:
                self.logger.error(f"Ollama embedding error: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.exception(f"Error embedding text with Ollama: {e}")
            return None
        
    def generate_text(self, prompt: str, chat_history: list = None,
                      max_output_tokens: int = None, temperature: float = None):

        if not self.generation_model_id:
            self.logger.error("Generation model for Ollama was not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        messages = []
        if chat_history:
            for item in chat_history:
                messages.append({
                    "role": item.get("role", "user"),
                    "content": item.get("text", "") or item.get("content", "")
                })

        messages.append({
            "role": "user",
            "content": self.process_text(prompt)
        })

        try:
            response = self.client.chat(
                model=self.generation_model_id,
                messages=messages,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_output_tokens
                }
            )
        except Exception as e:
            self.logger.error("Error during Ollama generation: %s", e)
            self.logger.exception("Traceback:")
            return None

        # Handle ollama-0.6.1 ChatResponse object
        # The response has a 'message' attribute that contains Message object with 'content'
        try:
            if hasattr(response, 'message'):
                # ollama-0.6.1 ChatResponse object
                message = response.message
                if hasattr(message, 'content'):
                    return message.content
            
            # Fallback for dict-like responses
            if isinstance(response, dict):
                if "message" in response and isinstance(response["message"], dict):
                    content = response["message"].get("content")
                    if content:
                        return content
                if "content" in response:
                    return response["content"]
        except Exception as e:
            self.logger.error("Error parsing Ollama response: %s", e)
            return None
        
        self.logger.error("Could not extract content from Ollama response")
        return None

    def embed_text(self, text: str, document_type: str = None):

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Ollama was not set")
            return None

        try:
            response = self.client.embeddings(
                model=self.embedding_model_id,
                prompt=self.process_text(text)
            )
        except Exception as e:
            self.logger.error("Error during Ollama embedding: %s", e)
            return None

        return response.get("embedding")

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "text": self.process_text(prompt)
        }
