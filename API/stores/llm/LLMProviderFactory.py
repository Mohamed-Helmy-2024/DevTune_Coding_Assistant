from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider,OllamaProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        # Create appropriate LLM provider implementation
        # Using logging for visibility
        import logging
        logging.getLogger(__name__).info(f"Creating provider: {provider}")
        if provider == LLMEnums.OPENAI.value:
            op = OpenAIProvider(
                api_key = self.config.OPENAI_API_KEY,
                api_url = self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
            if hasattr(self.config, 'OPENAI_MODEL'):
                op.set_generation_model(self.config.OPENAI_MODEL)
            if hasattr(self.config, 'OPENAI_EMBEDDING_MODEL'):
                # default embedding size for OpenAI embeddings is 1536, but can be overridden
                op.set_embedding_model(self.config.OPENAI_EMBEDDING_MODEL, embedding_size=getattr(self.config,'OPENAI_EMBEDDING_MODEL_SIZE', 1536))
            return op

        if provider == LLMEnums.COHERE.value:
            cp = CoHereProvider(
                api_key = self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DEFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DEFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DEFAULT_TEMPERATURE
            )
            cp.set_generation_model(self.config.COHERE_MODEL)
            if hasattr(self.config, 'COHERE_EMBEDDING_MODEL'):
                cp.set_embedding_model(self.config.COHERE_EMBEDDING_MODEL, embedding_size=getattr(self.config,'COHERE_EMBEDDING_MODEL_SIZE', 1024))
            return cp
        if provider == LLMEnums.OLLAMA.value:
            op = OllamaProvider()
            op.set_generation_model(self.config.OLLAMA_MODEL)
            if hasattr(self.config, 'OLLAMA_EMBEDDING_MODEL'):
                op.set_embedding_model(
                    self.config.OLLAMA_EMBEDDING_MODEL,
                    embedding_size=768
                )
            return op
        
        return None
