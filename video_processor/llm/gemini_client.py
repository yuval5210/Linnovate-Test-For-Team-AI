import os
import json
import re
import random
import google.generativeai as genai
import logging, sys
from typing import List, Dict
from PIL import Image
import time
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class GeminiClient:   
    def __init__(self, api_key: str):
        """Initialize Gemini client"""
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        self.api_key = api_key
        genai.configure(api_key=api_key)

        # Quota management
        self.daily_quota_limit = 100  # Free tier limit
        self.requests_made_today = 0
        self.last_reset_date = datetime.now().date()
        self.min_delay_between_requests = 3.0  # 3 seconds between requests
        self.last_request_time = 0
        
        # Initialize models
        try:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            raise

        self.__load_quota_usage()

    
    def __load_quota_usage(self):
        """Load quota usage from file if exists"""
        try:
            quota_file = '/tmp/gemini_quota.json'
            if os.path.exists(quota_file):
                with open(quota_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == str(datetime.now().date()):
                        self.requests_made_today = data.get('requests', 0)
                        logger.info(f"📊 Quota usage: {self.requests_made_today}/{self.daily_quota_limit}")
        except Exception as e:
            logger.warning(f"Could not load quota usage: {e}")
    

    def __save_quota_usage(self):
        """Save quota usage to file"""
        try:
            quota_file = '/tmp/gemini_quota.json'
            data = {
                'date': str(datetime.now().date()),
                'requests': self.requests_made_today
            }
            with open(quota_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save quota usage: {e}")

    
    def _check_and_manage_quota(self):
        """Check quota and implement rate limiting"""
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.requests_made_today = 0
            self.last_reset_date = current_date
            logger.info("🔄 Daily quota reset")
        
        # Check if we've exceeded daily quota
        if self.requests_made_today >= self.daily_quota_limit:
            logger.error(f"🚫 Daily quota exceeded ({self.requests_made_today}/{self.daily_quota_limit})")
            raise Exception("Daily Gemini API quota exceeded. Using fallback processing.")
        
        # Implement rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_delay_between_requests:
            sleep_time = self.min_delay_between_requests - time_since_last_request
            logger.debug(f"⏰ Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.requests_made_today += 1
        self.__save_quota_usage()
        
        logger.info(f"📈 API Request #{self.requests_made_today}/{self.daily_quota_limit}")
    

    def __validate_analysis(self, analysis: dict) -> bool:
        """Validate that analysis has required fields"""
        required_fields = ['importance_score', 'moment_description', 'category', 'keywords']
        
        for field in required_fields:
            if field not in analysis:
                return False
        
        # Check importance score is valid
        try:
            score = float(analysis['importance_score'])
            if not (0.0 <= score <= 1.0):
                return False
        except:
            return False
        
        # Check category is valid
        valid_categories = ['ACTION', 'DIALOGUE', 'SCENIC', 'EMOTIONAL', 'TECHNICAL', 'ENTERTAINMENT', 'TRANSITION', 'OTHER']
        if analysis['category'] not in valid_categories:
            return False
        
        # Check description exists
        if not analysis['moment_description'] or len(analysis['moment_description']) < 10:
            return False
        
        # Check keywords is a list
        if not isinstance(analysis['keywords'], list):
            return False
        
        return True
    

    def generate_highlight_description(self, image_path: str, 
                                     audio_text: str, timestamp: float) -> dict:
        """Generate comprehensive highlight description combining visual and audio"""
        try:
            try:
                self._check_and_manage_quota()
            except Exception as e:
                if "quota exceeded" in str(e).lower():
                    logger.warning("🔄 Using fallback processing due to quota limits")
                    return self.__generate_fallback_analysis(image_path, audio_text, timestamp)
                raise

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and validate image
            try:
                image = Image.open(image_path)
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return "Unable to load image for analysis"
            
            # Handle empty inputs gracefully
            audio_context = audio_text.strip() if audio_text else "No audio detected"
            
            prompt = f"""You are analyzing a video segment at {timestamp:.2f} seconds for comprehensive highlight extraction.

                    Analyze the image and audio text together to provide complete analysis:
                    **AUDIO TEXT**: "{audio_context}"

                    Provide analysis in JSON format:
                    {{
                    "importance_score": "importance score between 0.0 and 1.0",
                    "moment_description": "2-3 sentence description combining visual and audio",
                    "category": "ACTION/DIALOGUE/SCENIC/EMOTIONAL/TECHNICAL/ENTERTAINMENT/TRANSITION/OTHER",
                    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
                    }}

                    **Importance Score (0.0-1.0):**
                    Consider these factors:
                    1. Action intensity (movement, activity, dynamic content)
                    2. Visual interest (unique objects, scenes, compelling visuals)
                    3. Audio significance (speech, music, important sounds)
                    4. Potential viewer engagement (interesting, entertaining, informative)
                    5. Content uniqueness (distinctive moments vs. repetitive content)
                    Rate the importance on a scale of 0.0 to 1.0, where:
                    - 0.0-0.3: Low importance (static scenes, silence, repetitive content)
                    - 0.4-0.6: Medium importance (moderate activity, some interest)
                    - 0.7-1.0: High importance (key moments, significant events, engaging content)

                    **Categories:**
                    Categorize this video moment into one of these categories:
                    - ACTION: Physical activities, movement, sports, dynamic content
                    - DIALOGUE: People talking, conversations, interviews, presentations
                    - SCENIC: Beautiful views, landscapes, establishing shots, environments
                    - EMOTIONAL: Expressions of emotion, reactions, dramatic moments
                    - TECHNICAL: Demonstrations, explanations, how-to content, tutorials
                    - ENTERTAINMENT: Music, performances, comedy, artistic content
                    - TRANSITION: Scene changes, cuts, transitions between segments
                    - OTHER: Anything that doesn't clearly fit the above categories

                    **Keywords:**
                    Extract 5-8 relevant keywords from the moment_description text that would be useful for search.          
                    Return only the keywords, separated by commas.
                    Focus on:
                    - Important objects and subjects (people, animals, vehicles, etc.)
                    - Actions and activities (walking, speaking, demonstrating, etc.)
                    - Locations and settings (indoor, outdoor, office, park, etc.)
                    - Notable characteristics (colors, emotions, styles, etc.)
                    Example output: person, speaking, outdoor, gestures, presentation

                    **Moment Description:**
                    Generate a detailed description that:
                    1. Combines visual and audio information naturally
                    2. Highlights the most important and interesting aspects
                    3. Describes the main action or event taking place
                    4. Captures what makes this moment significant
                    5. Uses clear, searchable language suitable for video indexing
                    Keep the description informative but concise (2-3 sentences maximum).
                    Focus on what viewers would find most engaging or important.
                """
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content([prompt, image])
                    txt_response = response.text.strip()

                    analysis = json.loads(re.sub(r'^```json\s*|\s*```$', '', txt_response, flags=re.MULTILINE))

                    if self.__validate_analysis(analysis):
                        logger.debug(f"✅ Generated analysis at {timestamp:.1f}s")
                        return analysis
                    else:
                        logger.warning(f"⚠️ Invalid analysis format, using fallback")
                        return self.__generate_fallback_analysis(image_path, audio_text, timestamp)
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"🔧 JSON parsing failed attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower():
                        logger.error(f"🚫 Quota exceeded: {e}")
                        return self.__generate_fallback_analysis(image_path, audio_text, timestamp)
                    logger.warning(f"⚠️ Request failed attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
            
            logger.warning("🔄 All API attempts failed, using fallback analysis")
            return self.__generate_fallback_analysis(image_path, audio_text, timestamp)
            
        except Exception as e:
            logger.error(f"❌ Error generating highlight description at {timestamp:.2f}s: {str(e)}")
            return self.__generate_fallback_analysis(image_path, audio_text, timestamp)
    

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Gemini"""
        try:
            if not text or not text.strip():
                # Return zero vector for empty text
                return [0.0] * 768
            
            # Use Gemini's embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=text.strip(),
                task_type="retrieval_document"
            )
            
            embedding = result['embedding']
            logger.debug(f"Generated embedding of length {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return random vector as fallback (not ideal but prevents crashes)
            import random
            return [random.random() * 0.01 for _ in range(768)]
    

    def summarize_video_content(self, highlights: List[Dict]) -> str:
        """Generate a summary of video content based on highlights"""
        try:
            if not highlights:
                return "No highlights found in video"
            
            # Combine highlight descriptions, limiting total length
            descriptions = []
            total_length = 0
            max_length = 2000  # Limit input size
            
            for h in sorted(highlights, key=lambda x: x.get('importance_score', 0), reverse=True):
                desc = h.get('description', '')
                if total_length + len(desc) < max_length:
                    descriptions.append(desc)
                    total_length += len(desc)
            
            combined_text = ' '.join(descriptions)
            
            prompt = f"""
            Based on these video highlights, create a comprehensive summary of the video content:
            
            {combined_text}
            
            Generate a summary that:
            1. Captures the main themes and activities shown in the video
            2. Identifies key moments and significant events
            3. Describes the overall narrative or purpose of the content
            4. Mentions important subjects, locations, or topics covered
            5. Maintains a logical flow that reflects the video's progression
            
            Keep the summary concise but informative (3-5 sentences maximum).
            Write in a clear, engaging style suitable for video descriptions.
            """
            
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            if summary and len(summary) > 20:
                logger.info(f"Generated video summary: {summary[:100]}...")
                return summary
            else:
                # Fallback summary
                return f"Video containing {len(highlights)} highlighted moments with various activities and content."
            
        except Exception as e:
            logger.error(f"Error generating video summary: {str(e)}")
            return f"Video content with {len(highlights)} highlighted moments"


    def __generate_fallback_analysis(self, image_path: str, audio_text: str, timestamp: float) -> dict:
        """Generate analysis without API calls when quota is exceeded"""
        logger.info(f"🎭 Generating fallback analysis for {timestamp:.1f}s")
        
        # Basic analysis based on timestamp and audio
        frame_name = os.path.basename(image_path)
        
        # Generate description
        if audio_text and len(audio_text.strip()) > 0:
            description = f"Video moment at {timestamp:.1f}s with audio content: {audio_text[:100]}"
        else:
            description = f"Video frame captured at {timestamp:.1f}s showing visual content"
        
        # Heuristic importance scoring
        importance_score = self.__calculate_fallback_importance(audio_text, timestamp)
        
        # Determine category based on content
        category = self.__determine_fallback_category(audio_text, timestamp)
        
        # Extract simple keywords
        keywords = self.__extract_fallback_keywords(audio_text, description)
        
        return {
            'importance_score': importance_score,
            'moment_description': description,
            'category': category,
            'keywords': keywords
        }
    

    def __calculate_fallback_importance(self, audio_text: str, timestamp: float) -> float:
        """Calculate importance score without API"""
        score = 0.3  # Base score
        
        # Audio bonus
        if audio_text and len(audio_text.strip()) > 0:
            score += 0.2
            if len(audio_text.split()) > 5:
                score += 0.1
        
        # Action words bonus
        action_words = ['moving', 'walking', 'running', 'speaking', 'talking', 'opening', 'closing']
        if audio_text:
            audio_lower = audio_text.lower()
            action_count = sum(1 for word in action_words if word in audio_lower)
            score += min(0.2, action_count * 0.05)
        
        # Add small random variation
        score += random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, round(score, 2)))
    

    def __determine_fallback_category(self, audio_text: str, timestamp: float) -> str:
        """Determine category without API"""
        if not audio_text:
            return 'SCENIC'
        
        audio_lower = audio_text.lower()
        
        if any(word in audio_lower for word in ['speaking', 'talking', 'said', 'conversation']):
            return 'DIALOGUE'
        elif any(word in audio_lower for word in ['moving', 'walking', 'running', 'action']):
            return 'ACTION'
        elif any(word in audio_lower for word in ['music', 'singing', 'performance']):
            return 'ENTERTAINMENT'
        elif any(word in audio_lower for word in ['showing', 'demonstration', 'how']):
            return 'TECHNICAL'
        else:
            return 'OTHER'
    

    def __extract_fallback_keywords(self, audio_text: str, description: str) -> List[str]:
        """Extract keywords without API"""
        text = (audio_text or '') + ' ' + description
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter common words
        stop_words = {'the', 'and', 'with', 'this', 'that', 'video', 'frame', 'moment', 'content'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Return unique keywords, limited to 5
        unique_keywords = list(dict.fromkeys(keywords))[:5]
        
        # Ensure we have at least some keywords
        if not unique_keywords:
            unique_keywords = ['video', 'moment', 'content']
        
        return unique_keywords
    

    def get_quota_status(self) -> Dict[str, any]:
        """Get current quota usage status for monitoring"""
        return {
            'requests_made_today': self.requests_made_today,
            'daily_limit': self.daily_quota_limit,
            'remaining': max(0, self.daily_quota_limit - self.requests_made_today),
            'reset_date': str(self.last_reset_date),
            'quota_exceeded': self.requests_made_today >= self.daily_quota_limit
        }
    