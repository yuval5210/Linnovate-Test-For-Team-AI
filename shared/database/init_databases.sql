\echo 'Starting database initialization...'

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
\echo 'pgvector extension enabled'

-- Create videos table
CREATE TABLE IF NOT EXISTS videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    duration FLOAT NOT NULL,
    fps FLOAT NOT NULL,
    resolution VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary TEXT
);
\echo 'Videos table created'

-- Create highlights table
CREATE TABLE IF NOT EXISTS highlights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    timestamp FLOAT NOT NULL,
    description TEXT NOT NULL,
    embedding vector(768),
    importance_score FLOAT DEFAULT 0.0,
    category VARCHAR(50) DEFAULT 'OTHER',
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
\echo 'Highlights table created'

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_highlights_video_id ON highlights(video_id);
CREATE INDEX IF NOT EXISTS idx_highlights_timestamp ON highlights(timestamp);
CREATE INDEX IF NOT EXISTS idx_highlights_importance ON highlights(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_highlights_category ON highlights(category);
CREATE INDEX IF NOT EXISTS idx_videos_filename ON videos(filename);
CREATE INDEX IF NOT EXISTS idx_videos_created_at ON videos(created_at DESC);
\echo 'Basic indexes created'

-- Create vector similarity index
CREATE INDEX IF NOT EXISTS idx_highlights_embedding 
ON highlights USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
\echo 'Vector similarity index created'

-- Create a view for easy highlight retrieval with video information
CREATE OR REPLACE VIEW highlights_with_video AS
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
    v.duration as video_duration,
    v.fps,
    v.resolution,
    v.summary as video_summary
FROM highlights h
JOIN videos v ON h.video_id = v.id;
\echo 'Highlights view created'

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO "yuval_cohen";
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO "yuval_cohen";
GRANT ALL PRIVILEGES ON highlights_with_video TO "yuval_cohen";
\echo 'Permissions granted'

\echo 'Database initialization completed successfully!'

-- Display current database state
\echo 'Database ready for video processing!'
SELECT 
    schemaname,
    tablename 
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY tablename;