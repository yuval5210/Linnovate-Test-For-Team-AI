import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from typing import Optional
import logging, sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.connection = None
    

    def connect(self):
        """Create connection to PostgreSQL"""
        try:
            self.connection = psycopg2.connect(
                self.database_url,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            
            # Register vector type for pgvector operations
            register_vector(self.connection)
            
            logger.info("Database connection established successfully")
            
            # Test connection and verify schema
            self.__test_connection()
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {str(e)}")
            raise
    

    def disconnect(self):
        """
        Close database connection
        """
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    

    def get_cursor(self):
        """Get database cursor for queries"""
        if not self.connection:
            raise Exception("Database connection not initialized")
        return self.connection.cursor()
    

    def execute_query(self, query: str, params=None):
        """Execute SQL query and return results"""
        cursor = self.get_cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {query[:100]}... Error: {str(e)}")
            raise
        finally:
            cursor.close()
    

    def execute_single(self, query: str, params=None):
        """Execute a query and return single result"""
        cursor = self.get_cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Single query execution failed: {query[:100]}... Error: {str(e)}")
            raise
        finally:
            cursor.close()

    
    def execute_command(self, command: str, params=None):
        """Execute a command (INSERT, UPDATE, DELETE) and return affected row count"""
        cursor = self.get_cursor()
        try:
            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)
            return cursor.rowcount
        except Exception as e:
            logger.error(f"Command execution failed: {command[:100]}... Error: {str(e)}")
            raise
        finally:
            cursor.close()


    def __test_connection(self):
        """Test database connection and verify your existing schema"""
        try:
            cursor = self.get_cursor()
            
            # Simple connection test - just execute a query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # Don't check the result format, just that we got something
            if result is None:
                raise Exception("Database connection test failed - no result")
            
            logger.info(f"✅ Database connection successful")
            logger.info(f"Database result format: {type(result)}")
            
            # Check if pgvector extension exists
            try:
                cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                pgvector_result = cursor.fetchone()
                
                # Handle different result formats for pgvector check
                pgvector_exists = False
                if pgvector_result:
                    if isinstance(pgvector_result, (list, tuple)) and len(pgvector_result) > 0:
                        pgvector_exists = pgvector_result[0]
                    elif hasattr(pgvector_result, 'get'):
                        pgvector_exists = pgvector_result.get('exists', False) or pgvector_result.get(0, False)
                    else:
                        pgvector_exists = bool(pgvector_result)
                
                if pgvector_exists:
                    logger.info("✅ pgvector extension found and ready")
                else:
                    logger.warning("⚠️ pgvector extension not found - please run: CREATE EXTENSION vector;")
            except Exception as e:
                logger.warning(f"⚠️ Could not check pgvector status: {e}")
            
            # Check if your Step 1 tables exist
            try:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'videos'
                    )
                """)
                videos_result = cursor.fetchone()
                
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'highlights'
                    )
                """)
                highlights_result = cursor.fetchone()
                
                # Simple existence check without worrying about result format
                videos_exists = bool(videos_result) and videos_result != (False,) and videos_result != [False]
                highlights_exists = bool(highlights_result) and highlights_result != (False,) and highlights_result != [False]
                
                if videos_exists and highlights_exists:
                    logger.info("✅ Found your Step 1 tables: videos and highlights")
                    
                    # Check if tables have data
                    try:
                        cursor.execute("SELECT COUNT(*) FROM videos")
                        video_count_result = cursor.fetchone()
                        cursor.execute("SELECT COUNT(*) FROM highlights")  
                        highlight_count_result = cursor.fetchone()
                        
                        # Extract counts safely
                        video_count = 0
                        highlight_count = 0
                        
                        if video_count_result:
                            if isinstance(video_count_result, (list, tuple)):
                                video_count = video_count_result[0] if len(video_count_result) > 0 else 0
                            else:
                                video_count = int(video_count_result) if str(video_count_result).isdigit() else 0
                        
                        if highlight_count_result:
                            if isinstance(highlight_count_result, (list, tuple)):
                                highlight_count = highlight_count_result[0] if len(highlight_count_result) > 0 else 0
                            else:
                                highlight_count = int(highlight_count_result) if str(highlight_count_result).isdigit() else 0
                        
                        logger.info(f"📊 Database contains: {video_count} videos, {highlight_count} highlights")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Could not count table contents: {e}")
                    
                else:
                    logger.warning("⚠️ Missing required tables - please run your Step 1 video processing first")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not check table existence: {e}")
            
            cursor.close()
            logger.info("Database connection test completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {str(e)}")
            raise
    

    def test_vector_operations(self):
        """Test if vector operations work with your existing data"""
        try:
            cursor = self.get_cursor()
            
            # Check if we have any highlights with embeddings
            cursor.execute("SELECT COUNT(*) FROM highlights WHERE embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
            
            if embedding_count == 0:
                logger.info("No embeddings found in highlights table - vector test skipped")
                cursor.close()
                return
            
            # Test vector similarity query
            cursor.execute("""
                SELECT id, description, 
                       embedding <-> embedding as self_distance
                FROM highlights 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            if result and result['self_distance'] == 0.0:
                logger.info("✅ Vector operations working correctly")
            else:
                logger.warning("⚠️  Vector operations may not be working properly")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Vector operations test failed: {str(e)}")
            logger.warning("This might indicate pgvector is not properly installed")