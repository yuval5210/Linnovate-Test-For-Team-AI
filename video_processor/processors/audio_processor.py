import os
import sys
import tempfile
import subprocess
import logging
import speech_recognition as sr
from pathlib import Path
from pydub import AudioSegment

sys.path.append(str(Path(__file__).parent.parent))
from shared.config.settings import Settings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
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
        """Extract audio from video file"""
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
        """Convert audio file to text using speech recognition"""
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
        """Extract text from video audio using speech recognition"""
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


    def cleanup(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary audio files")
        except Exception as e:
            logger.error(f"Error cleaning up audio files: {str(e)}")