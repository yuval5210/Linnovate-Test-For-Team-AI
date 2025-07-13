from typing import List, Dict, Any, Optional
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import logging, sys

from database.connection import DatabaseConnection

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


class HighlightService:
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
    
    def search_similar_highlights(self, query_embedding: List[float], video_id: Optional[str] = None, limit: int = 5, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for highlights similar to the query using pgvector
        """
        try:
            # Build query with optional video_id filter using your existing schema
            base_query = """
                WITH scored_highlights AS (
                    SELECT 
                        h.id,
                        h.video_id,
                        h.timestamp,
                        h.description,
                        h.importance_score,
                        h.category,
                        h.keywords,
                        v.filename,
                        v.summary as video_summary,
                        h.embedding::vector <=> %s::vector as similarity_score
                    FROM highlights h
                    JOIN videos v ON h.video_id = v.id
                    WHERE h.embedding IS NOT NULL
                """
            
            params = [query_embedding]
            
            if video_id:
                base_query += " AND h.video_id = %s::uuid"
                params.append(video_id)
            
            base_query += """
                )
                SELECT * FROM scored_highlights
                WHERE similarity_score <= %s
                ORDER BY similarity_score ASC
                LIMIT %s
            """
            
            distance_threshold = 1.0 - similarity_threshold
            params.extend([distance_threshold, limit])
            
            # Execute query
            results = self.db_connection.execute_query(base_query, params)
            
            # Convert to list of dictionaries
            highlights = []
            for row in results:
                # Convert distance to similarity and apply threshold
                similarity = 1 - row['similarity_score']

                # Only include results above similarity threshold
                if similarity >= similarity_threshold:
                    highlights.append({
                        'id': str(row['id']),
                        'video_id': str(row['video_id']),
                        'timestamp': float(row['timestamp']),
                        'description': row['description'],
                        'importance_score': float(row['importance_score']) if row['importance_score'] else 0.0,
                        'category': row['category'],
                        'keywords': row['keywords'] if row['keywords'] else [],
                        'filename': row['filename'],
                        'video_summary': row['video_summary'],
                        'similarity': similarity
                    })
            
            return highlights
            
        except Exception as e:
            import traceback
            logger.error(f"❌ VECTOR SEARCH ERROR: {str(e)}")
            logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
            raise Exception(f"Error searching highlights: {str(e)}")
    

    def get_highlights_by_video(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Get all highlights for a specific video using your existing schema
        """
        try:
            query = """
                SELECT 
                    h.id,
                    h.video_id,
                    h.timestamp,
                    h.description,
                    h.importance_score,
                    h.category,
                    h.keywords,
                    h.created_at,
                    v.filename,
                    v.summary as video_summary,
                    v.duration
                FROM highlights h
                JOIN videos v ON h.video_id = v.id
                WHERE h.video_id = %s::uuid
                ORDER BY h.timestamp ASC
            """
            
            results = self.db_connection.execute_query(query, [video_id])
            
            return [
                {
                    'id': str(row['id']),
                    'video_id': str(row['video_id']),
                    'timestamp': float(row['timestamp']),
                    'description': row['description'],
                    'importance_score': float(row['importance_score']) if row['importance_score'] else 0.0,
                    'category': row['category'],
                    'keywords': row['keywords'] if row['keywords'] else [],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'filename': row['filename'],
                    'video_summary': row['video_summary'],
                    'duration': float(row['duration']) if row['duration'] else 0.0
                }
                for row in results
            ]
            
        except Exception as e:
            raise Exception(f"Error fetching highlights by video: {str(e)}")
    

    def get_video_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summary information for all videos using your existing schema
        """
        try:
            query = """
                SELECT 
                    v.id as video_id,
                    v.filename,
                    v.duration,
                    v.summary,
                    v.resolution,
                    v.fps,
                    COUNT(h.id) as total_highlights,
                    MIN(h.timestamp) as min_timestamp,
                    MAX(h.timestamp) as max_timestamp,
                    ARRAY_AGG(DISTINCT h.category) FILTER (WHERE h.category IS NOT NULL) as categories,
                    AVG(h.importance_score) as avg_importance
                FROM videos v
                LEFT JOIN highlights h ON v.id = h.video_id
                GROUP BY v.id, v.filename, v.duration, v.summary, v.resolution, v.fps
                ORDER BY v.created_at DESC
            """
            
            results = self.db_connection.execute_query(query)
            
            summaries = []
            for row in results:
                min_time = float(row['min_timestamp']) if row['min_timestamp'] else 0.0
                max_time = float(row['max_timestamp']) if row['max_timestamp'] else 0.0
                
                # Format duration range
                if min_time > 0 and max_time > 0:
                    duration_range = f"{int(min_time//60)}:{int(min_time%60):02d} - {int(max_time//60)}:{int(max_time%60):02d}"
                else:
                    duration_range = "No highlights"
                
                summaries.append({
                    'video_id': str(row['video_id']),
                    'filename': row['filename'],
                    'total_highlights': row['total_highlights'],
                    'duration': float(row['duration']) if row['duration'] else 0.0,
                    'duration_range': duration_range,
                    'description': row['summary'] if row['summary'] else f"Video: {row['filename']}",
                    'categories': row['categories'] if row['categories'] else [],
                    'avg_importance': float(row['avg_importance']) if row['avg_importance'] else 0.0,
                    'resolution': row['resolution'],
                    'fps': float(row['fps']) if row['fps'] else 0.0
                })
            
            return summaries
            
        except Exception as e:
            raise Exception(f"Error fetching video summaries: {str(e)}")
    
    
    def get_highlight_by_id(self, highlight_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific highlight by ID
        """
        try:
            query = """
                SELECT 
                    h.id,
                    h.video_id,
                    h.timestamp,
                    h.description,
                    h.importance_score,
                    h.category,
                    h.keywords,
                    h.embedding,
                    h.created_at,
                    v.filename,
                    v.summary as video_summary
                FROM highlights h
                JOIN videos v ON h.video_id = v.id
                WHERE h.id = %s::uuid
            """
            
            result = self.db_connection.execute_single(query, [highlight_id])
            
            if not result:
                return None
            
            return {
                'id': str(result['id']),
                'video_id': str(result['video_id']),
                'timestamp': float(result['timestamp']),
                'description': result['description'],
                'importance_score': float(result['importance_score']) if result['importance_score'] else 0.0,
                'category': result['category'],
                'keywords': result['keywords'] if result['keywords'] else [],
                'created_at': result['created_at'].isoformat() if result['created_at'] else None,
                'filename': result['filename'],
                'video_summary': result['video_summary']
            }
            
        except Exception as e:
            raise Exception(f"Error fetching highlight by ID: {str(e)}")
    

    def search_highlights_by_keywords(self, keywords: List[str], video_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search highlights using keyword matching as fallback
        """
        try:
            # Create search condition for keywords
            search_conditions = []
            params = []
            
            for keyword in keywords:
                search_conditions.append("(h.description ILIKE %s OR h.category ILIKE %s OR %s = ANY(h.keywords))")
                params.extend([f"%{keyword}%", f"%{keyword}%", keyword])
            
            base_query = """
                SELECT 
                    h.id,
                    h.video_id,
                    h.timestamp,
                    h.description,
                    h.importance_score,
                    h.category,
                    h.keywords,
                    h.created_at,
                    v.filename
                FROM highlights h
                JOIN videos v ON h.video_id = v.id
                WHERE ({})
            """.format(" OR ".join(search_conditions))
            
            if video_id:
                base_query += " AND h.video_id = %s::uuid"
                params.append(video_id)
            
            base_query += " ORDER BY h.importance_score DESC, h.timestamp ASC LIMIT %s"
            params.append(limit)
            
            results = self.db_connection.execute_query(base_query, params)
            
            return [
                {
                    'id': str(row['id']),
                    'video_id': str(row['video_id']),
                    'timestamp': float(row['timestamp']),
                    'description': row['description'],
                    'importance_score': float(row['importance_score']) if row['importance_score'] else 0.0,
                    'category': row['category'],
                    'keywords': row['keywords'] if row['keywords'] else [],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'filename': row['filename'],
                    'similarity': 0.8
                }
                for row in results
            ]
            
        except Exception as e:
            raise Exception(f"Error searching highlights by keywords: {str(e)}")