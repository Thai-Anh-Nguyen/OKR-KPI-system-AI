from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "OKR-KPI System AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Add other configuration variables here like model paths, DB URIs, etc.
    # PHOVERT_MODEL_PATH: str = "vinai/phobert-base"
    
    class Config:
        env_file = ".env"

settings = Settings()
