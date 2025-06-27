#!/usr/bin/env python3
"""
Utility script to convert existing MPEG-4 videos to H.264 format for better browser compatibility.
Run this script to fix existing annotated videos that can't be played in browsers.
"""

import os
import subprocess
import sqlite3
from pathlib import Path

def convert_video_to_h264(input_path: str, output_path: str) -> bool:
    """Convert video to H.264 format using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'medium',  # Balance between speed and compression
            '-crf', '23',  # Quality setting (lower = better quality)
            '-movflags', '+faststart',  # Enable web streaming
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully converted: {input_path}")
            return True
        else:
            print(f"❌ Failed to convert {input_path}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        return False

def main():
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found. Please install ffmpeg:")
        print("   macOS: brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        return

    # Connect to database
    db_path = 'hand_analysis.db'
    if not os.path.exists(db_path):
        print("❌ Database not found. Run the backend first to create the database.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all analyses with annotated videos
    cursor.execute('SELECT id, annotated_video_path FROM analyses WHERE annotated_video_path IS NOT NULL')
    analyses = cursor.fetchall()

    if not analyses:
        print("ℹ️  No annotated videos found in database.")
        return

    print(f"Found {len(analyses)} annotated videos to check/convert...")

    converted_count = 0
    for analysis_id, video_path in analyses:
        if not video_path or not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}")
            continue

        # Check if video is already H.264
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            codec = result.stdout.strip()
            
            if codec == 'h264':
                print(f"✅ Already H.264: {video_path}")
                continue
            
            print(f"🔄 Converting {codec} to H.264: {video_path}")
            
            # Create temporary output path
            temp_path = video_path + '.tmp.mp4'
            
            if convert_video_to_h264(video_path, temp_path):
                # Replace original with converted version
                os.replace(temp_path, video_path)
                converted_count += 1
            else:
                # Clean up temp file if conversion failed
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"❌ Error processing {video_path}: {e}")

    conn.close()
    print(f"\n🎉 Conversion complete! Converted {converted_count} videos to H.264 format.")
    print("Videos should now be compatible with web browsers.")

if __name__ == '__main__':
    main() 