from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND :str = None
    OPENAI_API_KEY: str
    OPENAI_API_URL: str = None
    OPENAI_MODEL: str
    OPENAI_EMBEDDING_MODEL: str

    COHERE_API_KEY: str
    COHERE_MODEL: str
    COHERE_EMBEDDING_MODEL: str

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str ="gpt-oss:120b-cloud"
    OLLAMA_EMBEDDING_MODEL: str = "embeddinggemma:latest"

    INPUT_DEFAULT_MAX_CHARACTERS: int = 2000
    GENERATION_DEFAULT_MAX_TOKENS: int = 8192
    GENERATION_DEFAULT_TEMPERATURE: float = 0.3
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings(): ## this makes any got by "get_settings().APP_NAME" 
    return Settings()