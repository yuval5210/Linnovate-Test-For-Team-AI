from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path

from api.chat_router import chat_router
from database.connection import DatabaseConnection

sys.path.append(str(Path(__file__).parent.parent))
from shared.config.settings import Settings

settings = Settings()

app = FastAPI(
    title="Video Highlights Chat API",
    description="Chat with your video highlights using semantic search",
    version="1.0.0"
)

# Initialize database connection (synchronous)
db_connection = DatabaseConnection(settings.database_url)
db_connection.connect()

# Store database connection in app state
app.state.db_connection = db_connection

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,  # Both React options
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Video Highlights Chat API is running"}

@app.get("/health")
def health_check():
    try:
        # Test database connection
        cursor = db_connection.get_cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        if result[0] == 1:
            return {"status": "healthy", "database": "connected"}
        else:
            return {"status": "unhealthy", "database": "failed"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )