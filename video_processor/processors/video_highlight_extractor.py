import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

from processors.video_processor import VideoProcessor
from processors.audio_processor import AudioProcessor
from llm.gemini_client import GeminiClient
from database.database_manager import DatabaseManager

sys.path.append(str(Path(__file__).parent.parent))
from shared.config.settings import Settings

logger = logging.getLogger(__name__)


class VideoHighlightExtractor:
    def __init__(self):
        self.settings = Settings()
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.llm_client = GeminiClient(api_key=self.settings.gemini_api_key)
        self.db_manager = DatabaseManager(database_url=self.settings.database_url)

        try:
            quota_status = self.llm_client.get_quota_status()
            logger.info(f"📊 API Quota: {quota_status['requests_made_today']}/{quota_status['daily_limit']}")
        except:
            logger.info("📊 Quota tracking not available")
        

    def process_video(self, video_path: str, video_name: Optional[str] = None) -> str:
        """
        Process a video file and extract highlights
        
        Args:
            video_path: Path to the video file
            video_name: Optional name for the video
            
        Returns:
            video_id: ID of the processed video in database
        """
        try:
            logger.info(f"Starting video processing for: {video_path}")
            
            # Validate video file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            filename = video_name or Path(video_path).name
            
            # Check if video already exists in database
            if self.db_manager.video_exists(filename):
                existing_video = self.db_manager.get_video_by_filename(filename)
                logger.info(f"Video {filename} already exists in database")
                
                # Check if it has highlights
                highlights = self.get_video_highlights(existing_video['id'])
                if highlights:
                    logger.info(f"Video already has {len(highlights)} highlights")
                    return str(existing_video['id'])
                else:
                    logger.info("Video exists but has no highlights, processing...")
                    return self.__process_video_content(video_path, existing_video['id'])
            
            # Extract video metadata
            video_info = self.video_processor.get_video_info(video_path)
            logger.info(f"Video info: {video_info}")
            
            # Create video record in database
            video_id = self.db_manager.create_video_record(
                filename=filename,
                duration=video_info['duration'],
                fps=video_info['fps'],
                resolution=f"{video_info['width']}x{video_info['height']}"
            )
            
            # Process the video
            return self.__process_video_content(video_path, video_id)
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    

    def __process_video_content(self, video_path: str, video_id: str) -> str:
        """Core video processing logic"""
        try:
            # Extract frames at key intervals
            frames = self.video_processor.extract_frames(video_path, interval=5)
            logger.info(f"Extracted {len(frames)} frames")
            
            # Extract audio and convert to text
            audio_text = self.audio_processor.extract_audio_text(video_path)
            logger.info(f"Extracted audio text: {len(audio_text)} characters")
            
            highlights_created = 0
            
            # Process each frame with LLM
            for i, (timestamp, frame_path) in enumerate(frames):
                try:                    
                    # Get relevant audio text for this timestamp
                    relevant_audio = self.__get_audio_for_timestamp(
                        audio_text, timestamp, window=10
                    )
                    
                    # Generate comprehensive description
                    frame_analysis = self.llm_client.generate_highlight_description(
                        image_path=frame_path,
                        audio_text=relevant_audio,
                        timestamp=timestamp
                    )

                    full_description = frame_analysis['moment_description']
                    
                    # Check if this is an important moment
                    importance_score = float(frame_analysis['importance_score'])
                    
                    if importance_score > self.settings.importance_threshold:
                        # Generate embedding for similarity search
                        embedding = self.llm_client.generate_embedding(full_description)
                        
                        # Get category and keywords
                        category = frame_analysis['category']
                        keywords = frame_analysis['keywords']
                        
                        # Save highlight to database
                        highlight_id = self.db_manager.create_highlight_record(
                            video_id=video_id,
                            timestamp=timestamp,
                            description=full_description,
                            embedding=embedding,
                            importance_score=importance_score,
                            category=category,
                            keywords=keywords
                        )
                        
                        highlights_created += 1
                        logger.info(f"Saved highlight {highlight_id} at {timestamp}s (score: {importance_score:.3f})")
                    
                    # Clean up frame file
                    os.remove(frame_path)
                    
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {str(e)}")
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                    continue
            
            # Generate and save video summary
            if highlights_created > 0:
                highlights = self.get_video_highlights(video_id)
                summary = self.llm_client.summarize_video_content(highlights)
                self.db_manager.update_video_summary(video_id, summary)
            
            logger.info(f"Completed processing video: {video_id} ({highlights_created} highlights created)")
            return video_id
            
        except Exception as e:
            logger.error(f"Error processing video content: {str(e)}")
            raise


    def __get_audio_for_timestamp(self, audio_text: str, timestamp: float, window: int = 10) -> str:
        """Get relevant audio text for a given timestamp"""
        if not audio_text:
            return ""
        
        words = audio_text.split()
        if not words:
            return ""
        
        words_per_second = len(words) / 60
        
        # Calculate word range for the timestamp window
        start_word = max(0, int((timestamp - window/2) * words_per_second))
        end_word = min(len(words), int((timestamp + window/2) * words_per_second))
        
        return ' '.join(words[start_word:end_word])
    

    def search_highlights(self, query: str, limit: int = 5) -> List[dict]:
        try:
            query_embedding = self.llm_client.generate_embedding(query)
            results = self.db_manager.search_similar_highlights(
                query_embedding, 
                limit,
                self.settings.similarity_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Error searching highlights: {str(e)}")
            return []
    

    def get_video_highlights(self, video_id: str) -> List[dict]:
        try:
            highlights = self.db_manager.get_video_highlights(video_id)
            return highlights
        except Exception as e:
            logger.error(f"Error retrieving highlights for video {video_id}: {str(e)}")
            return []


    def cleanup(self):
        try:
            self.video_processor.cleanup()
            self.audio_processor.cleanup()
            self.db_manager.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")