# 🎬 Video Highlights with LLM-Based Chat System

A comprehensive system that extracts intelligent highlights from videos using LLM analysis and provides an interactive React-based chat interface to query the processed content.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Step 1: Video Processing](#step-1-video-processing)
- [Step 2: Interactive Chat](#step-2-interactive-chat)
- [Browser Compatibility](#browser-compatibility)
- [Configuration](#configuration)
- [Technical Details](#technical-details)

## 🎯 Overview

This system consists of two main components:

### **Step 1: Video Processor with LLM-Based Highlight Extraction**
- Processes video files (`.mp4`, `.mov`) automatically
- Extracts visual frames and audio transcription using OpenCV and SpeechRecognition
- Uses Google Gemini LLM to identify important moments and generate descriptions
- Generates detailed descriptions, categories, keywords, and importance scores
- Stores highlights in PostgreSQL with pgvector for semantic similarity search
- Creates vector embeddings for each highlight for intelligent search capabilities

### **Step 2: Interactive Chat About Video Highlights**
- **React-based frontend** for natural language queries (as required)
- FastAPI backend for highlight search and retrieval using vector similarity
- Semantic search using Gemini embeddings and pgvector
- Real-time chat interface with contextual responses and confidence scores

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Chat API       │    │  Video Processor│
│   (React/HTML)  │◄──►│   (FastAPI)      │    │   (Python)      │
│   Port: 3000    │    │   Port: 8000     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                         │
                                ▼                         ▼
                       ┌─────────────────────────────────────────┐
                       │        PostgreSQL + pgvector            │
                       │         Port: 5432                      │
                       │   ┌─────────────┐ ┌─────────────────┐   │
                       │   │   Videos    │ │   Highlights    │   │
                       │   │   Table     │ │   w/ Embeddings │   │
                       │   └─────────────┘ └─────────────────┘   │
                       └─────────────────────────────────────────┘
```

## ✨ Features

### **Video Processing (Step 1)**
- **Intelligent Frame Extraction**: Processes frames at configurable intervals (default: 5s)
- **Audio Transcription**: Speech-to-text conversion with noise handling
- **LLM Analysis**: Google Gemini for moment importance and comprehensive descriptions
- **Smart Categorization**: ACTION, DIALOGUE, SCENIC, EMOTIONAL, TECHNICAL, ENTERTAINMENT, etc.
- **Keyword Extraction**: Automatic tagging for better searchability
- **Vector Embeddings**: 768-dimensional embeddings for semantic similarity search
- **Importance Scoring**: 0.0-1.0 scale with configurable thresholds
- **Quota Management**: Automatic fallback when API limits are reached

### **Chat Interface (Step 2)**
- **React Frontend**: Modern interactive interface built with React hooks
- **Natural Language Queries**: Ask questions like "What happened with the car?"
- **Semantic Search**: Find relevant moments using vector similarity matching
- **Real-time Results**: Instant responses with highlighted moments and timestamps
- **Video Filtering**: Search within specific videos or across all content
- **Rich Responses**: Detailed descriptions, categories, confidence scores, and similarity ratings
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Prerequisites

- **Docker & Docker Compose**: Latest version
- **Google AI Studio API Key**: Free key from [https://aistudio.google.com/](https://aistudio.google.com/)
- **Video Files**: MP4 or MOV format, 30 seconds to 1.5 minutes duration (as specified)
- **Modern Browser**: Chrome, Firefox, Safari, or Edge with JavaScript enabled

## 🚀 Quick Start

### **1. Clone and Setup**

```bash
# Clone the repository
git clone <your-repository-url>
cd <project-directory>

# Create environment file
cp .env.example .env

# Edit .env with your API key
nano .env
# Add: GEMINI_API_KEY=your_api_key_here
```

### **2. Add Your Videos**

```bash
# Copy your video files in the videos directory (30s-1.5min as required)
cp /path/to/your/sample1.mp4 videos/
cp /path/to/your/sample2.mp4 videos/
```

### **3. Choose Your Deployment Option**

You have two deployment options:

#### **Option A: Complete System (Main docker-compose.yml)**
```bash
# Start all services (automatic video processing + chat system)
docker-compose up

# This will:
# 1. Start PostgreSQL with pgvector
# 2. Automatically process all videos in ./videos directory
# 3. Start chat API backend (FastAPI)  
# 4. Start React frontend

# Wait for processing to complete, then access:
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

#### **Option B: Step1 Only (Separate Processing)**
```bash
# Step 1: Video Processing Only (using docker-compose.step1.yml)
docker-compose -f docker-compose.step1.yml up -d

# For manual demo/control:
docker exec -it video_processor_demo bash
python main.py  # Process all videos
```

### **4. Access the Application**

- **Frontend (React)**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Database**: localhost:5432 (if needed for inspection)

## 🎬 Step 1: Video Processing

You have two approaches for video processing:

### **Approach A: Automatic Processing (Main docker-compose.yml)**

```bash
# Start the complete system with automatic video processing
docker-compose up postgres video_processor

# Monitor processing logs
docker-compose logs -f video_processor

# This will automatically process all videos in ./videos directory
```

### **Approach B: Manual Processing (docker-compose.step1.yml)**

**Perfect for demos and manual control:**

```bash
# Start Step 1 services only (isolated environment)
docker-compose -f docker-compose.step1.yml up -d

# Access the video processor container for interactive demo
docker exec -it video_processor_demo bash

# Inside the container:
python main.py                    # Process all videos automatically

# Exit the container when done
exit

# Clean up step1 environment before proceeding to step2
docker-compose -f docker-compose.step1.yml down
```

### **Key Differences Between Approaches**

| Feature | Main docker-compose.yml | docker-compose.step1.yml |
|---------|------------------------|---------------------------|
| **Processing** | Automatic on startup | Manual/Interactive control |
| **Container Name** | `video_processor_app` | `video_processor_demo` |
| **Database** | `postgres_pgvector` | `video_highlights_postgres_step1` |
| **Use Case** | Production/Complete system | Demos/Development/Testing |
| **Auto Process** | `AUTO_PROCESS=true` | `AUTO_PROCESS=false` |
| **Command** | `python main.py` | `tail -f /dev/null` (manual) |

### **Expected Processing Output**

```
🚀 Starting Video Highlight Extraction
Processing video: ./videos/video_1.mp4
Processed video: video_1_uuid
📸 Extracted 12 frames (every 5s)
🎤 Extracted 1,250 characters of audio
✨ Saved highlight at 15.0s (score: 0.85) - [ACTION] Person exits vehicle
✨ Saved highlight at 32.5s (score: 0.72) - [DIALOGUE] Character speaking
✅ Successfully processed ./videos/video_1.mp4
🎯 Created 13 highlights

Processing video: ./videos/video_2.mp4
No speech detected in audio
Processed video: video_2_uuid
✅ Successfully processed ./videos/video_2.mp4
🎯 Created 9 highlights
```

## 💬 Step 2: Interactive Chat

### **Starting the Chat System**

Choose based on how you completed Step 1:

#### **If you used Main docker-compose.yml (Automatic Processing):**
```bash
# Chat services should already be starting automatically
# If not, start them manually:
docker-compose up chat_api frontend
```

#### **If you used docker-compose.step1.yml (Manual Processing):**
```bash
# First, stop the step1 environment to avoid conflicts
docker-compose -f docker-compose.step1.yml down

# Then start the chat system using main docker-compose.yml
# (It will connect to the existing processed data)
docker-compose up postgres chat_api frontend
```

#### **Complete System Restart (if needed):**
```bash
# Stop everything and start fresh
docker-compose down
docker-compose -f docker-compose.step1.yml down

# Start complete system
docker-compose up
```

### **Accessing the React Frontend**

```bash
# Open the chat interface
open http://localhost:3000
```

**⚠️ Important Browser Note:** If you encounter React loading errors, try opening the application in **private/incognito mode (Ctrl + Shift + N)** to avoid browser extension interference.

### **Using the React Chat Interface**

1. **Video Selection**: Choose specific videos from the sidebar dropdown or search all videos
2. **Ask Questions**: Type natural language queries in the chat input
3. **View Results**: See timestamped highlights with detailed descriptions
4. **Review Details**: Each result shows:
   - Video filename and timestamp
   - Similarity match percentage
   - Importance score
   - Category (ACTION, DIALOGUE, etc.)
   - Relevant keywords
   - Full description

### **Example Chat Queries**

```
💭 "What happened when the person got out of the car?"
🎬 "Show me action scenes"
🏠 "Find accidents in the kitchen"
⭐ "What are the most important moments?"
🎤 "Show me dialogue scenes"
🚗 "Find anything related to vehicles"
📍 "What happened around 30 seconds?"
🎯 "Show me high importance moments"
```

### **Chat Response Format**

The React interface displays structured responses with:

```json
{
  "answer": "I found 3 relevant moments for your question...",
  "relevant_highlights": [
    {
      "id": "uuid",
      "timestamp": 45.2,
      "description": "Person exits car in parking lot with luggage",
      "importance_score": 0.85,
      "category": "ACTION",
      "keywords": ["person", "car", "exit", "luggage"],
      "filename": "sample1.mp4",
      "similarity": 0.92
    }
  ],
  "confidence_score": 0.87,
  "total_highlights_found": 3
}
```

## 🌐 Browser Compatibility

### **Recommended Browsers**
- Chrome 90+ ✅
- Firefox 88+ ✅  
- Safari 14+ ✅
- Edge 90+ ✅

### **Known Issues & Solutions**

**React Loading Errors:**
```
If you see errors like "Minified React error #130" or "transformScriptTags.ts":
1. Try opening in private/incognito mode first
2. Disable browser extensions (especially React DevTools, live reload tools)
3. Clear browser cache and reload
4. Check if any ad blockers are interfering
```

**Extension Conflicts:**
- React Developer Tools may interfere in development
- Live Server/Hot Reload extensions can cause issues
- VS Code browser preview extensions may transform scripts

**Quick Fix:** Always test in private/incognito mode if you encounter frontend issues.


## ⚙️ Configuration

### **Environment Variables (.env)**

```bash
# Required - Get from https://aistudio.google.com/
GEMINI_API_KEY=your_google_ai_studio_key

# Database Configuration
DATABASE_URL=postgresql://yuval_cohen:password@postgres:5432/video_highlights

# Video Processing Settings
AUTO_PROCESS=true                 # Automatically process videos on startup
FRAME_EXTRACTION_INTERVAL=5       # Seconds between frame extractions
AUDIO_SEGMENT_DURATION=10         # Audio analysis window in seconds
IMPORTANCE_THRESHOLD=0.6          # Minimum score for highlights (0.0-1.0)
MAX_FRAME_WIDTH=1280             # Maximum frame width for processing

# Chat System Settings
API_PORT=8000                    # Backend API port
FRONTEND_PORT=3000               # React frontend port
SEARCH_SIMILARITY_THRESHOLD=0.5  # Minimum similarity for search results
MAX_HIGHLIGHTS_PER_RESPONSE=5    # Maximum results per query

# Performance Tuning
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
```

### **Processing Modes**

The system automatically adjusts based on Google AI Studio quota:

- **Full Mode**: Complete LLM processing with importance scoring
- **Conservative Mode**: Reduced API calls, key frames only
- **Fallback Mode**: Basic processing when quota is exceeded


## 🔧 Technical Details

### **Technology Stack**

**Backend Processing:**
- **Python 3.9+**: Core processing language with OOP structure
- **OpenCV**: Video frame extraction and computer vision
- **SpeechRecognition**: Audio transcription with multiple engine support
- **Google Gemini**: LLM for video analysis and embedding generation
- **PostgreSQL + pgvector**: Vector database for semantic similarity search
- **FastAPI**: High-performance web framework with automatic documentation

**Frontend (React as Required):**
- **React 18**: Modern functional components with hooks (useState, useEffect, useRef)
- **JSX**: Component-based architecture
- **Tailwind CSS**: Utility-first styling framework
- **Vanilla JavaScript**: No additional frameworks for maximum compatibility

**Infrastructure:**
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Production web server for frontend
- **Multi-stage Docker builds**: Optimized container images

### **Database Schema**

```sql
-- Videos table
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    duration FLOAT NOT NULL,
    fps FLOAT NOT NULL,
    resolution VARCHAR(50) NOT NULL,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Highlights table with vector embeddings
CREATE TABLE highlights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    timestamp FLOAT NOT NULL,
    description TEXT NOT NULL,
    embedding vector(768),  -- Gemini embedding dimension
    importance_score FLOAT DEFAULT 0.0,
    category VARCHAR(50) DEFAULT 'OTHER',
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector similarity index for fast search
CREATE INDEX idx_highlights_embedding 
ON highlights USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### **Video Processing Pipeline**

1. **Video Ingestion**: Validate format, extract metadata (duration, fps, resolution)
2. **Frame Extraction**: Sample frames at configured intervals using OpenCV
3. **Audio Processing**: Extract audio track and convert to text
4. **LLM Analysis**: Send frames + audio to Gemini for:
   - Importance scoring (0.0-1.0)
   - Detailed moment descriptions
   - Category classification
   - Keyword extraction
5. **Vector Embedding**: Generate 768-dimensional embeddings for semantic search
6. **Database Storage**: Store all data with proper indexing
7. **Video Summary**: Generate overall video summary from highlights

### **Chat Architecture**

**Frontend (React):**
- Modern React with functional components
- State management using useState hooks
- Real-time UI updates with useEffect
- Event handling for user interactions
- Responsive design with Tailwind CSS

**Backend (FastAPI):**
- RESTful API endpoints with automatic documentation
- Pydantic models for request/response validation
- Database connection pooling with proper error handling
- Vector similarity search using pgvector operators
- Structured response formatting with confidence scoring

**Search Algorithm:**
1. **Query Processing**: Generate embedding for user question using Gemini
2. **Vector Search**: Find similar highlights using cosine similarity
3. **Filtering**: Apply video-specific filters and similarity thresholds
4. **Ranking**: Sort by relevance score and importance
5. **Response Generation**: Format results with context and metadata

## 📊 Performance Considerations

- **API Quota Management**: Automatic rate limiting and fallback processing
- **Vector Search Optimization**: Efficient indexing with pgvector
- **Database Connection Pooling**: Optimized for concurrent requests
- **React Performance**: Efficient state updates and minimal re-renders
- **Container Resource Usage**: Reasonable memory and CPU allocation
- **Browser Compatibility**: Works across modern browsers without polyfills

## 🎯 Demo Script

Perfect for showcasing the complete system:

```bash
# 1. Show video processing capability
docker-compose up postgres video_processor
# Demonstrate: AI analysis, categorization, embedding generation

# 2. Start chat system
docker-compose up chat_api frontend

# 3. Demo React frontend (use private mode if needed)
open http://localhost:3000
# Show: Video selection, natural language queries, real-time results

# 4. Example queries to demonstrate:
# - "What are the most important moments?"
# - "Show me action scenes"  
# - "Find dialogue moments"
# - "What happened with [specific topic]?"

# 5. Show technical implementation
open http://localhost:8000/docs
# Demonstrate: API endpoints, request/response models

# 6. Show system logs and processing
docker-compose logs video_processor
docker-compose logs chat_api
```

## 🎯 Highlights

**Technical Requirements Met:**
- ✅ **Python backend** with OOP structure and proper separation of concerns
- ✅ **React frontend** using modern hooks and component architecture  
- ✅ **PostgreSQL + pgvector** for vector similarity search
- ✅ **LLM integration** with Google Gemini for analysis and embeddings
- ✅ **Docker containerization** for both services
- ✅ **RESTful API** with FastAPI and automatic documentation
- ✅ **Database-only responses** - chat pulls exclusively from stored highlights
- ✅ **Clean modular architecture** with proper error handling

---

**🚀 Ready to explore your videos with AI-powered semantic search using React and vector embeddings!**

For additional support, check the logs using `docker-compose logs [service-name]` or access the database directly for inspection.