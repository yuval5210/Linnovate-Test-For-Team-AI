import cv2
import os
import tempfile
from typing import List, Tuple, Dict
import logging, sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()


    def get_video_info(self, video_path: str) -> Dict:
        """Extract basic video information"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    

    def extract_frames(self, video_path: str, interval: int = 5) -> List[Tuple[float, str]]:
        """Extract frames from video at specified intervals, always including the last frame"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("Invalid FPS value")
            
            frame_interval = int(fps * interval)
            
            frames = []
            frame_num = 0
            last_frame_data = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_num / fps
                
                if frame_num % frame_interval == 0:                  
                    # Save frame to temporary file
                    frame_filename = f"frame_{timestamp:.2f}s.jpg"
                    frame_path = os.path.join(self.temp_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    frames.append((timestamp, frame_path))
                    
                    logger.debug(f"Extracted frame at {timestamp:.2f}s")
                
                frame_num += 1
            
            # Save last frame if it's different from the last saved interval frame
            if last_frame_data and len(frames) > 0:
                last_timestamp, last_frame = last_frame_data
                last_saved_timestamp = frames[-1][0]
                
                # Save last frame if it's more than 1 second after the last saved frame
                if last_timestamp - last_saved_timestamp > 1.0:
                    frame_filename = f"frame_{last_timestamp:.2f}s_last.jpg"
                    frame_path = os.path.join(self.temp_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, last_frame)
                    frames.append((last_timestamp, frame_path))
                    
                    logger.debug(f"Extracted last frame at {last_timestamp:.2f}s")

            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            if cap.isOpened():
                cap.release()
            logger.error(f"Error extracting frames: {str(e)}")
            raise


    def cleanup(self):
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary video files")
        except Exception as e:
            logger.error(f"Error cleaning up video files: {str(e)}")