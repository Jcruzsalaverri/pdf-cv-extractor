import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration management for PDF processing pipeline."""
    
    # API Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Multiple Gemini API Keys for rotation (to avoid rate limits)
    GEMINI_API_KEYS: list = []
    
    @classmethod
    def _load_gemini_keys(cls):
        """Load all available Gemini API keys from environment."""
        if cls.GEMINI_API_KEYS:
            return  # Already loaded
        
        keys = []
        
        # Option 1: Comma-separated list in GEMINI_API_KEYS (recommended)
        keys_env = os.getenv("GEMINI_API_KEYS")
        if keys_env:
            # Split by comma and strip whitespace
            keys = [key.strip() for key in keys_env.split(",") if key.strip()]
        else:
            # Option 2: Fallback to individual keys for backward compatibility
            # Add primary key if exists
            if cls.GEMINI_API_KEY:
                keys.append(cls.GEMINI_API_KEY)
            
            # Add numbered keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
            i = 1
            while True:
                key = os.getenv(f"GEMINI_API_KEY_{i}")
                if not key:
                    break
                keys.append(key.strip())
                i += 1
        
        cls.GEMINI_API_KEYS = keys
        if keys:
            print(f"🔑 Loaded {len(keys)} Gemini API key(s) for rotation")
    
    @classmethod
    def get_gemini_api_keys(cls) -> list:
        """Get all Gemini API keys."""
        cls._load_gemini_keys()
        return cls.GEMINI_API_KEYS
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Embedding Configuration (Local only - no API calls)
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-mpnet-base-v2")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "claude"
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    
    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
    
    # Vector Store Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        # Check LLM provider (only place we use API keys)
        if cls.LLM_PROVIDER == "gemini":
            keys = cls.get_gemini_api_keys()
            if not keys:
                raise ValueError(
                    "GEMINI_API_KEYS or GEMINI_API_KEY required for Gemini LLM. "
                    "Set LLM_PROVIDER=claude or provide API key(s)."
                )
        elif cls.LLM_PROVIDER == "claude" and not cls.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY required for Claude LLM. Set LLM_PROVIDER=gemini or provide API key."
            )
        
        return True
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of current configuration (without sensitive data)."""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "embedding_model": cls.LOCAL_EMBEDDING_MODEL,
            "embedding_batch_size": cls.EMBEDDING_BATCH_SIZE,
            "llm_provider": cls.LLM_PROVIDER,
            "api_keys_count": len(cls.get_gemini_api_keys())
        }
