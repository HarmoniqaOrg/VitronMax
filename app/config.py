"""Config module for loading and validating environment variables."""
from enum import Enum
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Supported environments for the application."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    ENV: Environment = Environment.DEVELOPMENT
    LOG_LEVEL: str = "INFO"

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None

    # Supabase
    SUPABASE_URL: Optional[str] = None
    SUPABASE_SERVICE_KEY: Optional[str] = None
    
    # Deployment
    FLY_API_TOKEN: Optional[str] = None
    
    # Model
    MODEL_PATH: str = "models/bbb_rf_v1_0.joblib"
    
    # Storage
    STORAGE_BUCKET_NAME: str = "vitronmax"

    @field_validator("ENV")
    def env_must_be_valid(cls, v: str) -> Environment:
        """Validate environment is one of the supported values."""
        try:
            return Environment(v.lower())
        except ValueError:
            valid_values = ", ".join([e.value for e in Environment])
            raise ValueError(f"ENV must be one of: {valid_values}")

    class Config:
        """Pydantic settings configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create a global instance of settings
settings = Settings()
