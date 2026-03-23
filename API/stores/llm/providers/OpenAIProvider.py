from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging

class OpenAIProvider(LLMInterface):
    def __init__(self, api_key: str, api_url: str = None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = None,
                      max_output_tokens: int = None, temperature: float = None):
        if not self.client:
            self.logger.error("OpenAI client was not initialized")
            return None

        if not self.generation_model_id:
            self.logger.error("OpenAI generation model not set")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature or self.default_generation_temperature

        messages = chat_history[:] if chat_history else []
        messages.append(self.construct_prompt(prompt, OpenAIEnums.USER.value))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_completion_tokens=max_output_tokens,
                temperature=temperature
            )

            if (not response or not response.choices
                    or not response.choices[0].message
                    or not response.choices[0].message.content):
                self.logger.error("Invalid OpenAI response format")
                return None

            return response.choices[0].message.content

        except Exception as e:
            self.logger.exception("Error while generating text with OpenAI: %s", e)
            return None

    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("OpenAI client was not initialized")
            return None

        if not self.embedding_model_id:
            self.logger.error("OpenAI embedding model not set")
            return None

        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=text
            )

            if not response or not response.data or not response.data[0].embedding:
                self.logger.error("Invalid OpenAI embedding response")
                return None

            return response.data[0].embedding

        except Exception as e:
            self.logger.exception("Error while embedding text with OpenAI: %s", e)
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
