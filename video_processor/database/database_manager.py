import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database operations with pgvector support"""
    
    def __init__(self, database_url: str):
        """
        Initialize database manager
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.connection = None
        self.connect()
        self.verify_database_setup()


    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise


    def close(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")


    def verify_database_setup(self):
        """Verify that required tables and extensions exist"""
        try:
            with self.connection.cursor() as cursor:
                # Check if pgvector extension exists
                cursor.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    );
                """)
                has_vector = cursor.fetchone()[0]
                
                if not has_vector:
                    logger.error("pgvector extension not found - database not properly initialized")
                    raise RuntimeError("Database missing pgvector extension")
                
                # Check if required tables exist
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('videos', 'highlights');
                """)
                
                existing_tables = [row[0] for row in cursor.fetchall()]
                required_tables = {'videos', 'highlights'}
                missing_tables = required_tables - set(existing_tables)
                
                if missing_tables:
                    logger.error(f"Missing required tables: {missing_tables}")
                    raise RuntimeError(f"Database missing tables: {missing_tables}")
                
                logger.info("Database setup verified successfully")
                
        except Exception as e:
            logger.error(f"Error verifying database setup: {str(e)}")
            raise
    
    
    def create_video_record(self, filename: str, duration: float, 
                          fps: float, resolution: str) -> str:
        """
        Create a new video record
        
        Args:
            filename: Video filename
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Video resolution (e.g., "1920x1080")
            
        Returns:
            Video ID
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO videos (filename, duration, fps, resolution)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id;
                """, (filename, duration, fps, resolution))
                
                video_id = cursor.fetchone()[0]
                logger.info(f"Created video record: {video_id}")
                return str(video_id)
                
        except psycopg2.IntegrityError as e:
            if "unique constraint" in str(e).lower():
                logger.warning(f"Video {filename} already exists in database")
                # Get existing video ID
                existing_video = self.get_video_by_filename(filename)
                if existing_video:
                    return str(existing_video['id'])
                else:
                    raise RuntimeError(f"Video {filename} exists but couldn't retrieve ID")
            else:
                logger.error(f"Database integrity error: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error creating video record: {str(e)}")
            raise
    

    def create_highlight_record(self, video_id: str, timestamp: float, 
                        description: str, embedding: List[float], 
                        importance_score: float = 0.0, 
                        category: str = 'OTHER',
                        keywords: List[str] = None) -> str:
        """
        Create a new highlight record
        
        Args:
            video_id: Video ID
            timestamp: Timestamp in seconds
            description: Highlight description
            embedding: Embedding vector
            importance_score: Importance score (0-1)
            category: Category label
            keywords: List of keywords
            
        Returns:
            Highlight ID
        """
        try:
            with self.connection.cursor() as cursor:
                embedding_vector = str(embedding) if embedding else None

                cursor.execute("""
                    INSERT INTO highlights 
                    (video_id, timestamp, description, embedding, importance_score, category, keywords)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (video_id, timestamp, description, embedding_vector, 
                      importance_score, category, keywords or []))
                
                highlight_id = cursor.fetchone()[0]
                logger.info(f"Created highlight: {highlight_id}")
                
                return str(highlight_id)
                
        except Exception as e:
            logger.error(f"Error creating highlight: {str(e)}")
            raise
    

    def get_video_highlights(self, video_id: str, limit: int = 100) -> List[Dict]:
        """
        Get all highlights for a video
        
        Args:
            video_id: Video ID
            limit: Maximum number of highlights to return
            
        Returns:
            List of highlight dictionaries
        """
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT h.*, v.filename, v.duration as video_duration
                    FROM highlights h
                    JOIN videos v ON h.video_id = v.id
                    WHERE h.video_id = %s
                    ORDER BY h.importance_score DESC, h.timestamp ASC
                    LIMIT %s;
                """, (video_id, limit))
                
                highlights = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(highlights)} highlights for video {video_id}")
                return highlights
                
        except Exception as e:
            logger.error(f"Error getting video highlights: {str(e)}")
            return []
    

    def search_similar_highlights(self, query_embedding: List[float], 
                                limit: int = 10, 
                                min_similarity: float = 0.5) -> List[Dict]:
        """
        Search for similar highlights using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar highlights with similarity scores
        """
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT h.*, v.filename, v.duration as video_duration,
                           1 - (h.embedding <=> %s) as similarity
                    FROM highlights h
                    JOIN videos v ON h.video_id = v.id
                    WHERE h.embedding IS NOT NULL
                    AND 1 - (h.embedding <=> %s) > %s
                    ORDER BY h.embedding <=> %s
                    LIMIT %s;
                """, (query_embedding, query_embedding, min_similarity, query_embedding, limit))
                
                results = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Found {len(results)} similar highlights")
                return results
                
        except Exception as e:
            logger.error(f"Error searching similar highlights: {str(e)}")
            return []
    

    def get_top_highlights(self, limit: int = 20) -> List[Dict]:
        """
        Get top highlights by importance score
        
        Args:
            limit: Maximum number of highlights to return
            
        Returns:
            List of top highlights
        """
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT h.*, v.filename, v.duration as video_duration
                    FROM highlights h
                    JOIN videos v ON h.video_id = v.id
                    ORDER BY h.importance_score DESC, h.created_at DESC
                    LIMIT %s;
                """, (limit,))
                
                highlights = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(highlights)} top highlights")
                return highlights
                
        except Exception as e:
            logger.error(f"Error getting top highlights: {str(e)}")
            return []
    
    
    def get_highlights_by_category(self, category: str, limit: int = 50) -> List[Dict]:
        """
        Get highlights by category
        
        Args:
            category: Category name
            limit: Maximum number of highlights
            
        Returns:
            List of highlights in the category
        """
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT h.*, v.filename, v.duration as video_duration
                    FROM highlights h
                    JOIN videos v ON h.video_id = v.id
                    WHERE h.category = %s
                    ORDER BY h.importance_score DESC
                    LIMIT %s;
                """, (category, limit))
                
                highlights = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(highlights)} highlights for category {category}")
                return highlights
                
        except Exception as e:
            logger.error(f"Error getting highlights by category: {str(e)}")
            return []
    

    def get_all_videos(self) -> List[Dict]:
        """Get all videos from database"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT v.*, 
                           COUNT(h.id) as highlight_count,
                           AVG(h.importance_score) as avg_importance
                    FROM videos v
                    LEFT JOIN highlights h ON v.id = h.video_id
                    GROUP BY v.id
                    ORDER BY v.created_at DESC;
                """)
                videos = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(videos)} videos from database")
                return videos
        except Exception as e:
            logger.error(f"Error getting all videos: {str(e)}")
            return []
    

    def get_video_by_filename(self, filename: str) -> Optional[Dict]:
        """Get video by filename"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM videos WHERE filename = %s;", (filename,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting video by filename: {str(e)}")
            return None
    

    def video_exists(self, filename: str) -> bool:
        """Check if video exists in database"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM videos WHERE filename = %s;", (filename,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking video existence: {str(e)}")
            return False
    

    def update_video_summary(self, video_id: str, summary: str):
        """
        Update video summary
        
        Args:
            video_id: Video ID
            summary: Video summary text
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE videos SET summary = %s WHERE id = %s;
                """, (summary, video_id))
                
                logger.info(f"Updated summary for video {video_id}")
                
        except Exception as e:
            logger.error(f"Error updating video summary: {str(e)}")
            raise
    

    def get_video_stats(self, video_id: str) -> Dict:
        """
        Get statistics for a video
        
        Args:
            video_id: Video ID
            
        Returns:
            Dictionary with video statistics
        """
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_highlights,
                        COALESCE(AVG(importance_score), 0) as avg_importance,
                        COALESCE(MAX(importance_score), 0) as max_importance,
                        COALESCE(MIN(timestamp), 0) as first_highlight,
                        COALESCE(MAX(timestamp), 0) as last_highlight,
                        COUNT(DISTINCT category) as unique_categories
                    FROM highlights
                    WHERE video_id = %s;
                """, (video_id,))
                
                stats = dict(cursor.fetchone())
                logger.info(f"Retrieved stats for video {video_id}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting video stats: {str(e)}")
            return {
                'total_highlights': 0,
                'avg_importance': 0.0,
                'max_importance': 0.0,
                'first_highlight': 0.0,
                'last_highlight': 0.0,
                'unique_categories': 0
            }
    

    def delete_video(self, video_id: str):
        """
        Delete a video and all its highlights
        
        Args:
            video_id: Video ID
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM videos WHERE id = %s;", (video_id,))
                logger.info(f"Deleted video {video_id}")
                
        except Exception as e:
            logger.error(f"Error deleting video: {str(e)}")
            raise
    
    

    def get_database_info(self) -> Dict:
        """Get general database information"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        (SELECT COUNT(*) FROM videos) as total_videos,
                        (SELECT COUNT(*) FROM highlights) as total_highlights,
                        (SELECT COUNT(*) FROM highlights WHERE embedding IS NOT NULL) as highlights_with_embeddings,
                        (SELECT version()) as postgresql_version;
                """)
                
                info = dict(cursor.fetchone())
                return info
                
        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            return {}