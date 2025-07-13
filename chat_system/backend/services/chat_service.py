from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import logging
import time
import google.generativeai as genai

from services.highlight_service import HighlightService

sys.path.append(str(Path(__file__).parent.parent))
from shared.config.settings import Settings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, highlight_service: HighlightService):
        self.highlight_service = highlight_service
        self.settings = Settings()
        
        # Initialize Gemini for embeddings (consistent with video processing)
        try:
            genai.configure(api_key=self.settings.gemini_api_key)
            self.genai = genai
            logger.info("✅ Chat service initialized with Gemini embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {e}")
            raise


    def process_question(self, question: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user question and return relevant highlights with structured answer
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing question: '{question[:50]}{'...' if len(question) > 50 else ''}'")

            if video_id:
                logger.info(f"📹 Filtering by video ID: {video_id}")
            
            # Step 1: Generate embedding for the question using Gemini
            question_embedding = self.__generate_question_embedding(question)
            
            if not question_embedding:
                return self.__create_error_response("Failed to generate question embedding")
            
            # Step 2: Search for relevant highlights
            relevant_highlights = self.highlight_service.search_similar_highlights(
                query_embedding=question_embedding,
                video_id=video_id,
                limit=self.settings.max_highlights_per_response,
                similarity_threshold=self.settings.similarity_threshold
            )
            
            if not relevant_highlights:
                return self.__create_no_results_response(question, video_id)
            
            # Step 3: Create structured response
            structured_answer = self.__create_structured_response(question, relevant_highlights)
            confidence = self.__calculate_confidence(relevant_highlights)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"✅ Found {len(relevant_highlights)} highlights (confidence: {confidence:.2f}, time: {processing_time}ms)")
            
            return {
                "answer": structured_answer,
                "relevant_highlights": relevant_highlights,
                "confidence_score": confidence,
                "processing_time_ms": processing_time,
                "total_highlights_found": len(relevant_highlights)
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {str(e)}")
            return self.__create_error_response(f"Error processing question: {str(e)}")
    

    def __generate_question_embedding(self, question: str) -> Optional[List[float]]:
        """
        Generate embedding for user question using Gemini
        Same model and approach as used in video processing for consistency
        """
        try:
            if not question or not question.strip():
                logger.warning("⚠️ Empty question provided")
                return None
            
            # Use same embedding model as video processing
            result = self.genai.embed_content(
                model="models/embedding-001",
                content=question.strip(),
                task_type="retrieval_query"  # Optimized for search queries
            )
            
            embedding = result['embedding']
            logger.debug(f"🔗 Generated question embedding of length {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate question embedding: {e}")
            # Try alternative approach with shorter question
            try:
                if len(question) > 100:
                    short_question = question[:100] + "..."
                    result = self.genai.embed_content(
                        model="models/embedding-001",
                        content=short_question,
                        task_type="retrieval_query"
                    )
                    return result['embedding']
            except Exception as e2:
                logger.error(f"❌ Fallback embedding also failed: {e2}")
            
            return None
        

    def __create_structured_response(self, question: str, highlights: List[Dict]) -> str:
        """
        Create a structured response based on matching highlights from your schema
        """
        if not highlights:
            return f"I couldn't find any relevant highlights for your question: '{question}'"
        
        # Sort highlights by timestamp for chronological order
        sorted_highlights = sorted(highlights, key=lambda x: x.get('timestamp', 0))
        
        # Analyze question type to customize response style
        question_lower = question.lower()
        
        # Build response sections
        response_parts = []
        
        # Add personalized context about what was found
        if len(highlights) == 1:
            if any(word in question_lower for word in ['when', 'what time']):
                response_parts.append(f"Based on your question about timing, I found 1 relevant moment:")
            elif any(word in question_lower for word in ['what', 'describe', 'tell me']):
                response_parts.append(f"Here's what I found regarding '{question}':")
            else:
                response_parts.append(f"I found 1 relevant moment for your question:")
        else:
            if any(word in question_lower for word in ['show me', 'find', 'list']):
                response_parts.append(f"Here are {len(highlights)} moments I found for '{question}':")
            elif any(word in question_lower for word in ['when', 'what time']):
                response_parts.append(f"I found {len(highlights)} time-related moments for your question:")
            else:
                response_parts.append(f"Based on your question '{question}', I found {len(highlights)} relevant moments:")
        
        # Group by categories if available
        categories = {}
        for highlight in sorted_highlights:
            category = highlight.get('category', 'OTHER')
            if category not in categories:
                categories[category] = []
            categories[category].append(highlight)
        
        # Add each highlight with enhanced information
        for i, highlight in enumerate(sorted_highlights, 1):
            timestamp = highlight.get('timestamp', 0)
            description = highlight.get('description', 'No description available')
            filename = highlight.get('filename', 'unknown video')
            category = highlight.get('category', 'OTHER')
            importance = highlight.get('importance_score', 0.0)
            similarity = highlight.get('similarity_score', 0.0)
            keywords = highlight.get('keywords', [])
            
            # Format timestamp
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes}:{seconds:02d}"
            
            # Build highlight description with question context
            if 'when' in question_lower or 'what time' in question_lower:
                highlight_info = f"\n{i}. **{time_str}** in {filename}"
            else:
                highlight_info = f"\n{i}. At {time_str} in {filename}"
            
            # Add category and importance if meaningful
            if category != 'OTHER' or importance > 0 or similarity > 0:
                details = []
            if category != 'OTHER':
                details.append(f"{category.title()}")
            if importance > 0:
                details.append(f"Importance: {importance:.1f}")
            if similarity > 0:
                details.append(f"Match: {similarity*100:.0f}%")
            
            if details:
                highlight_info += f" [{', '.join(details)}]"
            
            highlight_info += f": {description}"
            
            # Add keywords if available and relevant to question
            if keywords:
                # Check if any keywords relate to the question
                question_words = set(question_lower.split())
                relevant_keywords = [k for k in keywords[:3] if k.lower() in question_words]
                
                if relevant_keywords:
                    highlight_info += f" (Related keywords: {', '.join(relevant_keywords)})"
                elif keywords:
                    highlight_info += f" (Keywords: {', '.join(keywords[:3])})"
            
            response_parts.append(highlight_info)
        
        # Add question-specific summary
        if len(highlights) > 1:
            if any(word in question_lower for word in ['most important', 'best', 'top']):
                avg_importance = sum(h.get('importance_score', 0) for h in highlights) / len(highlights)
                response_parts.append(
                    f"\nThese {len(highlights)} moments have an average importance score of {avg_importance:.1f}, "
                    f"which should help answer your question about the most significant events."
                )
            elif 'category' in question_lower or any(cat.lower() in question_lower for cat in categories.keys()):
                category_summary = ", ".join([f"{len(cat_highlights)} {cat.lower()}" 
                                            for cat, cat_highlights in categories.items()])
                response_parts.append(f"\nRegarding your question about categories, these highlights include: {category_summary}.")
            else:
                response_parts.append(
                    f"\nThese {len(highlights)} moments should provide a comprehensive answer to your question."
                )
        
        # Add follow-up suggestion based on question type
        if any(word in question_lower for word in ['what happened after', 'then what', 'next']):
            response_parts.append(f"\n💡 *Tip: You can ask about specific times or events for more details.*")
        elif any(word in question_lower for word in ['show me more', 'other', 'different']):
            response_parts.append(f"\n💡 *Tip: Try asking about specific categories like 'action scenes' or 'accidents'.*")
        
        return "".join(response_parts)
    

    def __calculate_confidence(self, highlights: List[Dict]) -> float:
        """
        Calculate confidence score based on similarity scores
        """
        if not highlights:
            return 0.0
        
        # Get similarity scores (assuming they're returned from the search)
        similarities = [h.get('similarity_score', 0.0) for h in highlights]
        
        # Calculate average similarity as confidence
        avg_similarity = sum(similarities) / len(similarities)

        # Boost confidence for multiple good results
        result_count_factor = min(1.0, len(highlights) / 3.0)
        
        importance_scores = [h.get('importance_score', 0.0) for h in highlights]
        avg_importance = sum(importance_scores) / len(importance_scores) if importance_scores else 0.0

        confidence = (
            avg_similarity * 0.6 + # Primary factor: how well queries match
            result_count_factor * 0.2 + # Secondary: number of results
            avg_importance * 0.2 # Tertiary: quality of moments
        )
        
        return max(0.0, min(1.0, round(confidence, 2)))
    

    def __create_no_results_response(self, question: str, video_id: Optional[str]) -> Dict[str, Any]:
        """Create response when no highlights are found"""
        if video_id:
            answer = f"I couldn't find any relevant highlights for '{question}' in the specified video. Try asking about different topics or removing the video filter to search all videos."
        else:
            answer = f"I couldn't find any relevant highlights for '{question}'. Try rephrasing your question or asking about different topics like 'action scenes', 'dialogue', or 'important moments'."
        
        return {
            "answer": answer,
            "relevant_highlights": [],
            "confidence_score": 0.0,
            "processing_time_ms": 0,
            "total_highlights_found": 0
        }
    

    def __create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create response for error cases"""
        return {
            "answer": f"Sorry, I encountered an issue: {error_message}. Please try again or rephrase your question.",
            "relevant_highlights": [],
            "confidence_score": 0.0,
            "processing_time_ms": 0,
            "total_highlights_found": 0
        }
    

    def get_conversation_context(self, video_id: str = None) -> Dict[str, Any]:
        """
        Get context information for the conversation
        """
        try:
            if video_id:
                highlights = self.highlight_service.get_highlights_by_video(video_id)
                return {
                    "video_id": video_id,
                    "total_highlights": len(highlights),
                    "available_topics": self.__extract_topics(highlights),
                    "categories": self.__get_categories(highlights),
                    "time_range": self.__get_time_range(highlights)
                }
            else:
                summaries = self.highlight_service.get_video_summaries()
                return {
                    "available_videos": len(summaries),
                    "video_summaries": summaries,
                    "total_highlights": sum(s.get('total_highlights', 0) for s in summaries)
                }
        except Exception as e:
            logger.error(f"❌ Error getting conversation context: {str(e)}")
            return {"error": str(e)}
    

    def __extract_topics(self, highlights: List[Dict]) -> List[str]:
        """
        Extract common topics/themes from highlights
        """
        if not highlights:
            return []
        
        # Collect keywords from all highlights
        all_keywords = []
        for highlight in highlights:
            keywords = highlight.get('keywords', [])
            all_keywords.extend(keywords)
        
        # Count keyword frequency
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        
        # Return most common keywords as topics
        return [keyword for keyword, count in keyword_counts.most_common(10)]
    

    def __get_categories(self, highlights: List[Dict]) -> Dict[str, int]:
        """Get category distribution from highlights"""
        from collections import Counter
        categories = [h.get('category', 'OTHER') for h in highlights]
        return dict(Counter(categories))
    

    def __get_time_range(self, highlights: List[Dict]) -> Dict[str, float]:
        """Get time range covered by highlights"""
        if not highlights:
            return {"start": 0.0, "end": 0.0}
        
        timestamps = [h.get('timestamp', 0.0) for h in highlights]
        return {
            "start": min(timestamps),
            "end": max(timestamps)
        }
    

    def suggest_questions(self, video_id: Optional[str] = None) -> List[str]:
        """
        Suggest relevant questions based on available highlights
        """
        try:
            if video_id:
                highlights = self.highlight_service.get_highlights_by_video(video_id)
            else:
                # Get sample highlights from all videos
                summaries = self.highlight_service.get_video_summaries()
                highlights = []
                for summary in summaries[:3]:  # Sample from first 3 videos
                    video_highlights = self.highlight_service.get_highlights_by_video(summary['video_id'])
                    highlights.extend(video_highlights[:2])  # 2 highlights per video
            
            if not highlights:
                return [
                    "What are the most important moments?",
                    "Show me action scenes",
                    "Find dialogue moments",
                    "What happened in this video?"
                ]
            
            # Generate suggestions based on available content
            suggestions = []
            categories = self.__get_categories(highlights)
            topics = self.__extract_topics(highlights)
            
            # Category-based suggestions
            if 'ACTION' in categories:
                suggestions.append("Show me action scenes")
            if 'DIALOGUE' in categories:
                suggestions.append("Find dialogue moments")
            if 'SCENIC' in categories:
                suggestions.append("Show me scenic views")
            
            # Topic-based suggestions
            if topics:
                for topic in topics[:3]:
                    suggestions.append(f"What happened with {topic}?")
            
            # General suggestions
            suggestions.extend([
                "What are the most important moments?",
                "Tell me about this video",
                "Find interesting highlights"
            ])
            
            return suggestions[:8]  # Return up to 8 suggestions
            
        except Exception as e:
            logger.error(f"❌ Error generating suggestions: {str(e)}")
            return [
                "What are the most important moments?",
                "Show me action scenes",
                "Find dialogue moments",
                "What happened in this video?"
            ]