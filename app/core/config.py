from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "OKR-KPI System AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Model paths
    PHOBERT_MODEL_NAME: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    KNN_MODEL_DIR: str = "./models"
    
    # Database (for RAG/future features)
    DATABASE_URL: Optional[str] = None
    
    # LLM Configuration (for RAG)
    LLM_PROVIDER: str = "openai"  # openai or anthropic
    LLM_MODEL: str = "gpt-4"
    LLM_API_KEY: Optional[str] = None
    
    # Performance tuning
    THREAD_POOL_WORKERS: int = 4
    MAX_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True


settings = Settings()
