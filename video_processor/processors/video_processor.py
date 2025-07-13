import cv2
import os
import tempfile
from typing import List, Tuple, Dict
import logging

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
        """
        Extract frames from video at specified intervals, always including the last frame
        
        Args:
            video_path: Path to video file
            interval: Interval in seconds between frames
            
        Returns:
            List of (timestamp, frame_path) tuples
        """
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
    

    def extract_scene_changes(self, video_path: str, threshold: float = 0.3) -> List[float]:
        """
        Detect scene changes in video
        
        Args:
            video_path: Path to video file
            threshold: Threshold for scene change detection
            
        Returns:
            List of timestamps where scene changes occur
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("Invalid FPS value")
            scene_changes = []
            
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return scene_changes
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_num = 1
            
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate histogram difference
                hist_prev = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
                hist_curr = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
                
                # Compare histograms
                correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
                
                if correlation < (1 - threshold):
                    timestamp = frame_num / fps
                    scene_changes.append(timestamp)
                    logger.debug(f"Scene change detected at {timestamp:.2f}s")
                
                prev_gray = curr_gray
                frame_num += 1
            
            cap.release()
            logger.info(f"Detected {len(scene_changes)} scene changes")
            return scene_changes
            
        except Exception as e:
            if cap.isOpened():
                cap.release()
            logger.error(f"Error detecting scene changes: {str(e)}")
            raise
    

    def extract_motion_moments(self, video_path: str, threshold: int = 1000) -> List[float]:
        """
        Extract moments with significant motion
        
        Args:
            video_path: Path to video file
            threshold: Motion threshold
            
        Returns:
            List of timestamps with significant motion
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("Invalid FPS value")
            motion_moments = []
            
            ret, prev_frame = cap.read()
            if not ret:
                cap.release()
                return motion_moments
            
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_num = 1
            
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_score = cv2.countNonZero(diff)
                
                if motion_score > threshold:
                    timestamp = frame_num / fps
                    motion_moments.append(timestamp)
                    logger.debug(f"Motion detected at {timestamp:.2f}s (score: {motion_score})")
                
                prev_gray = curr_gray
                frame_num += 1
            
            cap.release()
            logger.info(f"Detected {len(motion_moments)} motion moments")
            return motion_moments
            
        except Exception as e:
            if cap.isOpened():
                cap.release()
            logger.error(f"Error detecting motion: {str(e)}")
            raise
    

    def get_video_thumbnail(self, video_path: str, timestamp: float = 0) -> str:
        """
        Extract a thumbnail from video at specified timestamp
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp to extract thumbnail (default: 0)
            
        Returns:
            Path to thumbnail image
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("Invalid FPS value")
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                # If timestamp is invalid, get first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                
            if ret:
                thumbnail_path = os.path.join(self.temp_dir, f"thumbnail_{timestamp:.2f}s.jpg")
                
                # Resize for thumbnail
                height, width = frame.shape[:2]
                if width > 320:
                    scale = 320 / width
                    new_width = 320
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                cv2.imwrite(thumbnail_path, frame)
                cap.release()
                return thumbnail_path
            else:
                cap.release()
                raise ValueError("Could not extract thumbnail")
                
        except Exception as e:
            if cap.isOpened():
                cap.release()
            logger.error(f"Error extracting thumbnail: {str(e)}")
            raise


    def cleanup(self):
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary video files")
        except Exception as e:
            logger.error(f"Error cleaning up video files: {str(e)}")