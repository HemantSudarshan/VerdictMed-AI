"""
CDSS Configuration Module
Centralized settings management using Pydantic
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cdss"
    postgres_user: str = "cdss_user"
    postgres_password: str = "secure_password"
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "secure_password"
    
    # Weaviate
    weaviate_url: str = "http://localhost:8080"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # LLM
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:70b"
    
    # Security
    secret_key: str = "your-secret-key-here"
    encryption_key: str = "your-encryption-key-here"
    
    # Safety thresholds
    min_confidence_threshold: float = 0.55
    escalation_threshold: float = 0.70
    max_uncertainty: float = 0.15
    
    # Feature flags
    enable_image_analysis: bool = True
    enable_caching: bool = True
    debug_mode: bool = False
    
    # Monitoring
    prometheus_port: int = 9090
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to avoid re-reading environment on every call.
    """
    return Settings()
