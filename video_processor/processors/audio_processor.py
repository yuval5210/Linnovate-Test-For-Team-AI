import os
import sys
import tempfile
import subprocess
import logging
import speech_recognition as sr
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment

sys.path.append(str(Path(__file__).parent.parent))
from shared.config.settings import Settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.speech_recognizer = sr.Recognizer()
        self.temp_dir = tempfile.mkdtemp()
        self.settings = Settings()

        # Configure recognizer for better accuracy
        self.speech_recognizer.energy_threshold = 300
        self.speech_recognizer.dynamic_energy_threshold = True
        self.speech_recognizer.pause_threshold = 0.8


    def __extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        try:
            audio_path = os.path.join(self.temp_dir, "audio.wav")
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg stderr: {result.stderr}")
                # Try alternative extraction method
                return self.__extract_audio_with_pydub(video_path)
            
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise RuntimeError("Audio extraction produced empty file")
            
            logger.info(f"Extracted audio to: {audio_path}")
            return audio_path
            
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            raise RuntimeError("Audio extraction timed out")
        except FileNotFoundError:
            logger.warning("FFmpeg not found, trying alternative method")
            return self.__extract_audio_with_pydub(video_path)
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise

    
    def __extract_audio_with_pydub(self, video_path: str) -> str:
        """Alternative audio extraction using pydub"""
        try:
            audio_path = os.path.join(self.temp_dir, "audio_pydub.wav")
            
            # Load video and extract audio
            audio = AudioSegment.from_file(video_path)
            
            # Convert to mono and set sample rate
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Export as WAV
            audio.export(audio_path, format="wav")
            
            logger.info(f"Extracted audio using pydub: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error with pydub audio extraction: {str(e)}")
            raise
    
    
    def __audio_to_text(self, audio_path: str, silence_duration: float = 1.5) -> str:
        """
        Convert audio file to text using speech recognition
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Check if file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                logger.warning("Audio file is empty or doesn't exist")
                return ""
            
            audio_with_silence = self.__add_leading_silence(audio_path, silence_duration=silence_duration)

            # Load audio file
            with sr.AudioFile(audio_with_silence) as source:
                # Adjust for ambient noise
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.speech_recognizer.record(source)
            
            # Try multiple recognition services in order of preference
            text = self.__speech_recognition(audio)
            
            if text:
                logger.info(f"Transcribed audio: {len(text)} characters")
                return text.strip()
            else:
                logger.warning("No speech detected in audio")
                return ""
            
        except Exception as e:
            logger.error(f"Error converting audio to text: {str(e)}")
            return ""


    def __speech_recognition(self, audio) -> str:
        """Speech recognition services"""
        # Try Google Speech Recognition (free tier)
        try:
            text = self.speech_recognizer.recognize_google(audio)
            if text:
                logger.debug("Successfully used Google Speech Recognition")
                return text
        except sr.UnknownValueError:
            logger.debug("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            logger.warning(f"Google Speech Recognition request error: {e}")
        except Exception as e:
            logger.warning(f"Google Speech Recognition error: {e}")
        
        return ""

    
    def __add_leading_silence(self, audio_path: str, silence_duration: float = 1.0) -> str:
        """Add silence to the beginning of audio for better recognition"""
        try:            
            # Load original audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Create silence (same sample rate and channels as original)
            silence_ms = int(silence_duration * 1000)  # Convert to milliseconds
            silence = AudioSegment.silent(
                duration=silence_ms,
                frame_rate=audio.frame_rate
            )
            
            # Add silence to the beginning
            audio_with_silence = silence + audio
            
            # Save the modified audio
            modified_path = audio_path.replace('.wav', '_with_silence.wav')
            audio_with_silence.export(modified_path, format="wav")
            
            logger.info(f"✅ Added {silence_duration}s of silence to beginning")
            logger.info(f"📊 Original: {len(audio)/1000:.2f}s → Modified: {len(audio_with_silence)/1000:.2f}s")
            
            return modified_path
            
        except Exception as e:
            logger.error(f"Error adding silence: {e}")
            return audio_path  # Return original if failed
       
    
    def extract_audio_text(self, video_path: str) -> str:
        """
        Extract text from video audio using speech recognition
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcribed text from audio
        """
        try:
            # Extract audio from video
            audio_path = self.__extract_audio_from_video(video_path)
            
            # Convert to text
            text = self.__audio_to_text(audio_path, silence_duration=self.settings.slience_duration)
            
             # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting audio text: {str(e)}")
            return ""
    

    def extract_audio_segments(self, video_path: str, segment_duration: int = 10) -> List[Dict]:
        """
        Extract audio segments with timestamps
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of audio segments with timestamps and text
        """
        try:
            # Extract full audio
            audio_path = self.__extract_audio_from_video(video_path)
            
            # Load audio with pydub
            audio = AudioSegment.from_wav(audio_path)
            total_duration = len(audio) / 1000  # Convert to seconds
            
            segments = []
            segment_start = 0
            
            while segment_start < total_duration:
                segment_end = min(segment_start + segment_duration, total_duration)
                
                # Extract segment
                start_ms = int(segment_start * 1000)
                end_ms = int(segment_end * 1000)
                segment_audio = audio[start_ms:end_ms]

                # Skip very short segments
                if len(segment_audio) < 1000:  # Less than 1 second
                    break
                
                # Save segment to temporary file
                segment_path = os.path.join(self.temp_dir, f"segment_{segment_start:.1f}.wav")
                segment_audio.export(segment_path, format="wav")
                
                # Convert segment to text
                segment_text = self.__audio_to_text(segment_path)
                
                segments.append({
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'text': segment_text,
                    'duration': segment_end - segment_start,
                    'has_speech': bool(segment_text.strip())
                })
                
                # Clean up segment file
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                
                segment_start = segment_end
                
                logger.debug(f"Processed segment {segment_start:.1f}-{segment_end:.1f}s")
            
            # Clean up main audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            speech_segments = sum(1 for s in segments if s['has_speech'])
            logger.info(f"Extracted {len(segments)} audio segments ({speech_segments} with speech)")
            return segments
            
        except Exception as e:
            logger.error(f"Error extracting audio segments: {str(e)}")
            return []
    

    def detect_silence(self, video_path: str, silence_threshold: int = -40) -> List[Dict]:
        """
        Detect silent moments in audio
        
        Args:
            video_path: Path to video file
            silence_threshold: Silence threshold in dB
            
        Returns:
            List of silent periods with timestamps
        """
        try:
            # Extract audio
            audio_path = self.__extract_audio_from_video(video_path)
            
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Detect silence
            silence_ranges = []
            chunk_size = 1000  # 1 second chunks
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                
                if chunk.dBFS < silence_threshold:
                    start_time = i / 1000.0
                    end_time = (i + len(chunk)) / 1000.0
                    
                    silence_ranges.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
            
            # Clean up
            os.remove(audio_path)
            
            logger.info(f"Detected {len(silence_ranges)} silent periods")
            return silence_ranges
            
        except Exception as e:
            logger.error(f"Error detecting silence: {str(e)}")
            return []
    

    def analyze_audio_features(self, video_path: str) -> Dict:
        """
        Analyze audio features like volume, pitch, etc.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of audio features
        """
        try:
            # Extract audio
            audio_path = self.__extract_audio_from_video(video_path)
            
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Calculate features
            features = {
                'duration': len(audio) / 1000.0,
                'max_volume': audio.max_dBFS,
                'rms_volume': audio.rms,
                'sample_rate': audio.frame_rate,
                'channels': audio.channels,
                'has_audio': len(audio) > 0 and audio.max_dBFS > -60  # Basic audio detection
            }
            
            # Clean up
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            logger.info(f"Analyzed audio features: duration={features['duration']:.1f}s, has_audio={features['has_audio']}")
            return features
            
        except Exception as e:
            logger.error(f"Error analyzing audio features: {str(e)}")
            return {
                'duration': 0,
                'max_volume': -60,
                'rms_volume': 0,
                'sample_rate': 16000,
                'channels': 1,
                'has_audio': False
            }
    

    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary audio files")
        except Exception as e:
            logger.error(f"Error cleaning up audio files: {str(e)}")