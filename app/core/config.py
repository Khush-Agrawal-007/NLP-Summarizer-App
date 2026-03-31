# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Model configuration
    MODEL_NAME: str = "facebook/bart-large-cnn"
    
    # Summarization parameters
    MIN_SUMMARY_LENGTH: int = 200
    MAX_SUMMARY_LENGTH: int = 400
    
    # Text processing
    MIN_TEXT_LENGTH: int = 50
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

settings = Settings()