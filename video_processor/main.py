import os
from processors.video_highlight_extractor import VideoHighlightExtractor


def main():
    """Main function for demonstration"""
    extractor = VideoHighlightExtractor()

    print("🚀 Starting Video Highlight Extraction")
    
    # Example usage
    video_files = [os.path.join('./videos', file_name) for file_name in os.listdir("./videos") if file_name.endswith('.mp4')]

    try:
        for video_file in video_files:
            print(f"Processing video: {video_file}")
            if os.path.exists(video_file):
                try:
                    video_id = extractor.process_video(video_file)
                    print(f"Processed video: {video_id}")
                    
                    # Display highlights
                    highlights = extractor.get_video_highlights(video_id)
                    print(f"✅ Successfully processed {video_file}")
                    print(f"🎯 Created {len(highlights)} highlights")
                    for highlight in highlights[:3]:  # Show top 3
                        print(f"  - {highlight['timestamp']}s: {highlight['description'][:100]}...")
                        
                except Exception as e:
                    print(f"❌ Error processing {video_file}: {str(e)}")
            else:
                print(f"❌ Video file not found: {video_file}")
    except KeyboardInterrupt:
        print("👋 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()