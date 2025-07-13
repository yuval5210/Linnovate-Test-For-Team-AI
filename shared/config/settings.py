import os
from dataclasses import dataclass

@dataclass
class Settings:
    """Shared settings for all services in the video highlights system"""
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "")
    
    # LLM
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    
    # Video Processing (Step 1)
    video_input_dir: str = os.getenv("VIDEO_INPUT_DIR", "./videos")
    frame_interval: int = int(os.getenv("FRAME_INTERVAL", "5"))  # seconds
    importance_threshold: float = float(os.getenv("IMPORTANCE_THRESHOLD", "0.6"))
    max_video_duration: int = int(os.getenv("MAX_VIDEO_DURATION", "90"))  # 1.5 minutes max
    slience_duration: float = float(os.getenv("SILENCE_DURATION", "1.0"))  # seconds
    
    # Chat System (Step 2)
    api_port: int = int(os.getenv("API_PORT", "8000"))
    frontend_port: int = int(os.getenv("FRONTEND_PORT", "3000"))
    max_highlights_per_response: int = int(os.getenv("MAX_HIGHLIGHTS_PER_RESPONSE", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
    
    
    def __post_init__(self):
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
    
    @property
    def cors_origins_list(self) -> list:
        """Convert CORS_ORIGINS string to list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]