from fastapi import APIRouter, HTTPException, Request, Depends
from typing import List, Dict, Any
from pydantic import BaseModel

from services.chat_service import ChatService
from services.highlight_service import HighlightService
from database.connection import DatabaseConnection

chat_router = APIRouter()

class ChatRequest(BaseModel):
    question: str
    video_id: str = None


class ChatResponse(BaseModel):
    answer: str
    relevant_highlights: List[Dict[str, Any]]
    confidence_score: float


class HighlightSummary(BaseModel):
    video_id: str
    total_highlights: int
    duration_range: str
    description: str


def get_db_connection(request: Request) -> DatabaseConnection:
    return request.app.state.db_connection


@chat_router.post("/chat", response_model=ChatResponse)
def chat_with_highlights(chat_request: ChatRequest, db_connection: DatabaseConnection = Depends(get_db_connection)):
    """
    Chat endpoint that searches video highlights based on user questions
    """
    try:
        # Initialize services
        highlight_service = HighlightService(db_connection)
        chat_service = ChatService(highlight_service)
        
        # Process the chat request
        response = chat_service.process_question(
            question=chat_request.question,
            video_id=chat_request.video_id
        )
        
        return ChatResponse(
            answer=response["answer"],
            relevant_highlights=response["relevant_highlights"],
            confidence_score=response["confidence_score"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@chat_router.get("/highlights/summary", response_model=List[HighlightSummary])
def get_highlights_summary(db_connection: DatabaseConnection = Depends(get_db_connection)):
    """
    Get summary of available video highlights
    """
    try:
        highlight_service = HighlightService(db_connection)
        summaries = highlight_service.get_video_summaries()
        
        return [
            HighlightSummary(
                video_id=summary["video_id"],
                total_highlights=summary["total_highlights"],
                duration_range=summary["duration_range"],
                description=summary["description"]
            )
            for summary in summaries
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching highlights summary: {str(e)}"
        )


@chat_router.get("/highlights/{video_id}")
def get_video_highlights(video_id: str, db_connection: DatabaseConnection = Depends(get_db_connection)):
    """
    Get all highlights for a specific video
    """
    try:
        highlight_service = HighlightService(db_connection)
        highlights = highlight_service.get_highlights_by_video(video_id)
        
        if not highlights:
            raise HTTPException(
                status_code=404,
                detail=f"No highlights found for video: {video_id}"
            )
        
        return {
            "video_id": video_id,
            "highlights": highlights,
            "total_count": len(highlights)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching video highlights: {str(e)}"
        )