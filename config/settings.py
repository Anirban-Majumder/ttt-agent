from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    
    # Database Configuration
    chroma_db_path: str = Field(default="./data/chroma_db", env="CHROMA_DB_PATH")
    
    # Agent Configuration
    agent_name: str = Field(default="TTT-Agent", env="AGENT_NAME")
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    planning_timeout: int = Field(default=30, env="PLANNING_TIMEOUT")
    execution_timeout: int = Field(default=120, env="EXECUTION_TIMEOUT")
    
    # UI Configuration
    gradio_port: int = Field(default=7860, env="GRADIO_PORT")
    gradio_host: str = Field(default="localhost", env="GRADIO_HOST")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./data/logs/agent.log", env="LOG_FILE")
    
    # Memory Configuration
    max_conversation_history: int = Field(default=100, env="MAX_CONVERSATION_HISTORY")
    max_task_memory: int = Field(default=50, env="MAX_TASK_MEMORY")
    memory_retrieval_k: int = Field(default=5, env="MEMORY_RETRIEVAL_K")
    
    # Tool Configuration
    auto_approve_safe_tools: bool = Field(default=False, env="AUTO_APPROVE_SAFE_TOOLS")
    require_confirmation_by_default: bool = Field(default=True, env="REQUIRE_CONFIRMATION_BY_DEFAULT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __post_init__(self):
        """Create necessary directories."""
        Path(self.chroma_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
