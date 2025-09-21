import os
import re
import cv2
import numpy as np
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import crop
import yt_dlp
from moviepy.config import change_settings

# Cloudinary configuration using environment variables
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "djm2brvkw"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "922121246486646"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "_u029saczuQhse5atn8ogTrVy3Q"),
    secure=True
)

# Configuration settings
OUTPUT_FOLDER = "generated_shorts"
MAX_SHORT_LENGTH = 60
MIN_SHORT_LENGTH = 30
SHORT_RATIO = (9, 16)
FONT_MAPPING = {
    'en': 'DejaVu-Sans',
    'hi': 'Noto-Sans-Devanagari'  # Use Noto Sans Devanagari for Hindi
}

# Try to configure ImageMagick with the correct binary for Render
try:
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/magick"})
    print("ImageMagick configured successfully at /usr/bin/magick.")
except Exception as e:
    print(f"Warning: Failed to configure ImageMagick: {e}. Text overlays may be skipped.")

# FastAPI app initialization
app = FastAPI(title="YouTube Short Generator API")

# Request model for FastAPI
class VideoRequest(BaseModel):
    youtube_url: str
    segment_method: str = "1"  # 1 for engagement-based, 2 for even segments

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces_in_frame(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

    def get_safe_text_position(self, clip, start_time, duration):
        try:
            sample_times = np.linspace(start_time, start_time + duration, num=min(5, int(duration)))
            face_positions = []
            
            for t in sample_times:
                frame = clip.get_frame(t)
                faces = self.detect_faces_in_frame(frame)
                
                for (x, y, w, h) in faces:
                    rel_x = (x + w/2) / clip.size[0]
                    rel_y = (y + h/2) / clip.size[1]
                    face_positions.append((rel_x, rel_y))
            
            if not face_positions:
                return 0.83
            
            avg_y = sum(pos[1] for pos in face_positions) / len(face_positions)
            text_y = max(0.7, min(0.9, avg_y - 0.15))
            return text_y
        except Exception as e:
            print(f"Error determining text position: {e}")
            return 0.83

class YouTubeDownloader:
    def extract_video_id(self, url: str) -> str:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        return match.group(1) if match else ""

    def get_video_info(self, video_url: str) -> dict:
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extract_flat': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}

    def download_video(self, video_id: str, output_folder: str) -> str:
        try:
            os.makedirs(output_folder, exist_ok=True)
            temp_path = os.path.join(output_folder, f"temp_{video_id}.mp4")
            
            ydl_opts = {
                'format': 'best[height<=720]',
                'outtmpl': temp_path,
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            return temp_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return ""

class TranscriptManager:
    def __init__(self):
        self.transcript = []
        self.language = 'en'

    def get_transcript(self, video_id: str) -> bool:
        try:
            # Try to get available transcript languages
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try auto-generated English first, then Hindi, then any available
            language_priority = ['en', 'hi']
            selected_transcript = None
            
            for lang in language_priority:
                try:
                    selected_transcript = transcript_list.find_transcript([lang])
                    break
                except NoTranscriptFound:
                    continue
            
            if not selected_transcript:
                # Fallback to any available transcript
                selected_transcript = transcript_list.find_transcript()
            
            # Convert transcript snippets to dictionary format for consistency
            self.transcript = [
                {
                    'start': snippet.start,
                    'duration': snippet.duration,
                    'text': snippet.text
                }
                for snippet in selected_transcript.fetch()
            ]
            self.language = selected_transcript.language_code
            print(f"Found transcript in {selected_transcript.language}")
            return True
            
        except TranscriptsDisabled:
            print("Transcripts are disabled for this video")
            return False
        except NoTranscriptFound:
            print("No transcript available for this video")
            return False
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return False

class VideoAnalyzer:
    def __init__(self, min_short_length: int, max_short_length: int):
        self.min_short_length = min_short_length
        self.max_short_length = max_short_length
        self.engagement_data = []

    def analyze_engagement(self, video_path: str, transcript: list, video_length: int) -> bool:
        try:
            if not transcript:
                print("No transcript available - creating segments of at least 30 seconds")
                segment_count = min(3, int(video_length // self.min_short_length))
                
                for i in range(segment_count):
                    start = i * (video_length / segment_count)
                    end = start + self.min_short_length
                    
                    if end > video_length:
                        end = video_length
                        start = max(0, end - self.min_short_length)
                    
                    self.engagement_data.append({
                        'start': start,
                        'end': end,
                        'score': 80
                    })
                return True

            segments = []
            current_segment = None
            
            for entry in transcript:
                start = entry['start']
                end = start + entry['duration']
                text = entry['text']
                word_count = len(text.split())
                density = word_count / entry['duration'] if entry['duration'] > 0 else 0
                
                if current_segment is None:
                    current_segment = {
                        'start': start,
                        'end': end,
                        'text': text,
                        'score': density * 10
                    }
                else:
                    potential_end = end
                    if potential_end - current_segment['start'] <= self.max_short_length:
                        current_segment['end'] = potential_end
                        current_segment['text'] += " " + text
                        current_segment['score'] += density * 10
                    else:
                        if current_segment['end'] - current_segment['start'] >= self.min_short_length:
                            segments.append(current_segment)
                        current_segment = {
                            'start': start,
                            'end': end,
                            'text': text,
                            'score': density * 10
                        }
            
            if current_segment and current_segment['end'] - current_segment['start'] >= self.min_short_length:
                segments.append(current_segment)

            segments.sort(key=lambda x: x['score'], reverse=True)
            self.engagement_data = segments[:3]
            
            if not self.engagement_data:
                print("No valid segments from transcript - using default segments")
                return self.analyze_engagement(video_path, [], video_length)
                
            return True
            
        except Exception as e:
            print(f"Error analyzing engagement: {e}")
            return False

    def create_even_segments(self, video_length: int) -> bool:
        try:
            self.engagement_data = []
            segment_count = int(video_length // self.min_short_length)
            
            for i in range(segment_count):
                start = i * self.min_short_length
                end = start + self.min_short_length
                
                if end > video_length:
                    end = video_length
                    start = max(0, end - self.min_short_length)
                
                self.engagement_data.append({
                    'start': start,
                    'end': end,
                    'score': 80
                })
            return True
        except Exception as e:
            print(f"Error creating even segments: {e}")
            return False

class VideoProcessor:
    def __init__(self, short_ratio: tuple, font_mapping: dict):
        self.short_ratio = short_ratio
        self.font_mapping = font_mapping
        self.imagemagick_available = self._check_imagemagick()

    def _check_imagemagick(self):
        """Check if ImageMagick is available by attempting to create a simple text clip."""
        try:
            test_clip = TextClip("test", fontsize=12, color='white', font='DejaVu-Sans')
            test_clip.close()
            print("ImageMagick is available for text overlays.")
            return True
        except Exception as e:
            print(f"ImageMagick not available: {e}. Text overlays will be skipped.")
            return False

    def convert_to_shorts_format(self, clip):
        try:
            width, height = clip.size
            target_ratio = self.short_ratio[0] / self.short_ratio[1]
            
            if width / height > target_ratio:
                new_width = height * target_ratio
                x_center = width / 2
                return crop(
                    clip, 
                    x1=x_center - new_width/2, 
                    x2=x_center + new_width/2
                )
            else:
                new_height = width / target_ratio
                y_center = height / 2
                return crop(
                    clip,
                    y1=y_center - new_height/2,
                    y2=y_center + new_height/2
                )
        except Exception as e:
            print(f"Error converting to shorts format: {e}")
            return clip

    def add_transcript_overlay(self, clip, start_time: float, transcript: list, language: str, face_detector):
        try:
            if not transcript:
                print("No transcript available, skipping text overlay.")
                return clip
            if not self.imagemagick_available:
                print("ImageMagick not available, skipping text overlay.")
                return clip
            
            clip_end = start_time + clip.duration
            relevant_segments = [
                seg for seg in transcript
                if seg['start'] < clip_end and (seg['start'] + seg['duration']) > start_time
            ]
            
            font = self.font_mapping.get(language, 'DejaVu-Sans')
            print(f"Using font: {font} for language: {language}")
            safe_padding = int(clip.size[1] * 0.08)
            text_y_position = clip.size[1] - safe_padding

            subtitles = []
            for seg in relevant_segments:
                seg_start = max(0, seg['start'] - start_time)
                seg_end = min(clip.duration, (seg['start'] + seg['duration']) - start_time)
                
                try:
                    txt_clip = TextClip(
                        seg['text'],
                        fontsize=12,
                        color='#FF9933',
                        bg_color='black',
                        size=(clip.size[0], None),
                        method='caption',
                        font=font,
                        align='center'
                    ).set_position(
                        ("center", text_y_position - 40)
                    ).set_start(seg_start).set_duration(seg_end - seg_start)
                    
                    subtitles.append(txt_clip)
                    print(f"Added text clip for segment: {seg['text']} from {seg_start:.1f}s to {seg_end:.1f}s")
                except Exception as e:
                    print(f"Error creating text clip for segment '{seg['text']}': {e}")
                    continue
            
            if subtitles:
                print(f"Applying {len(subtitles)} subtitle clips to video.")
                return CompositeVideoClip([clip] + subtitles)
            print("No valid subtitles created, returning original clip.")
            return clip
        except Exception as e:
            print(f"Error adding transcript overlay: {e}")
            return clip

    def generate_short(self, video_path: str, start_time: float, end_time: float, clip_num: int,
                      video_id: str, video_length: float, transcript: list, language: str, face_detector) -> dict:
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found at {video_path}")
                return {"path": "", "url": ""}

            actual_start = max(0, start_time)
            actual_end = min(end_time, video_length)
            
            min_short_length = 30
            if actual_end - actual_start < min_short_length:
                actual_end = min(actual_start + min_short_length, video_length)
                if actual_end - actual_start < min_short_length:
                    actual_start = max(0, actual_end - min_short_length)
            
            print(f"Creating short from {actual_start:.1f}s to {actual_end:.1f}s (duration: {actual_end-actual_start:.1f}s)")
            
            with VideoFileClip(video_path) as clip:
                subclip = clip.subclip(actual_start, actual_end)
                vertical_clip = self.convert_to_shorts_format(subclip)
                final_clip = self.add_transcript_overlay(vertical_clip, actual_start, transcript, language, face_detector)
                
                output_folder = os.path.dirname(video_path)
                local_path = os.path.join(output_folder, f"short_{video_id}_{clip_num}.mp4")
                final_clip.write_videofile(
                    local_path,
                    codec='libx264',
                    audio_codec='aac',
                    threads=4,
                    fps=24,
                    preset='fast',
                    logger=None
                )
                
                # Upload to Cloudinary
                print(f"Uploading short {clip_num} to Cloudinary")
                upload_result = cloudinary.uploader.upload(
                    local_path,
                    resource_type="video",
                    public_id=f"shorts/short_{video_id}_{clip_num}",
                    overwrite=True
                )
                
                # Clean up local file
                try:
                    os.remove(local_path)
                    print(f"Cleaned up local file: {local_path}")
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {local_path}: {e}")
                
                return {"path": local_path, "url": upload_result["secure_url"]}
        except Exception as e:
            print(f"Error generating short: {e}")
            return {"path": "", "url": ""}

class YouTubeShortGenerator:
    def __init__(self):
        self.video_url = ""
        self.video_id = ""
        self.video_title = "YouTube Video"
        self.video_length = 0
        self.output_folder = OUTPUT_FOLDER
        self.max_short_length = MAX_SHORT_LENGTH
        self.min_short_length = MIN_SHORT_LENGTH
        self.short_ratio = SHORT_RATIO
        self.font_mapping = FONT_MAPPING
        self.downloader = YouTubeDownloader()
        self.transcript_manager = TranscriptManager()
        self.video_analyzer = VideoAnalyzer(self.min_short_length, self.max_short_length)
        self.video_processor = VideoProcessor(self.short_ratio, self.font_mapping)
        self.face_detector = FaceDetector()

    async def process_video(self, youtube_url: str, segment_method: str) -> dict:
        try:
            self.video_url = youtube_url
            video_id = self.downloader.extract_video_id(youtube_url)
            if not video_id:
                raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            self.video_id = video_id
            
            video_info = self.downloader.get_video_info(self.video_url)
            if not video_info:
                raise HTTPException(status_code=400, detail="Could not fetch video info")
            self.video_title = video_info.get('title', 'YouTube Video')
            self.video_length = video_info.get('duration', 0)
            
            # Fetch transcript with auto language detection
            if not self.transcript_manager.get_transcript(self.video_id):
                print("Proceeding without transcript")
            
            # Download video
            video_path = self.downloader.download_video(self.video_id, self.output_folder)
            if not video_path:
                raise HTTPException(status_code=500, detail="Failed to download video")
            
            # Analyze video or create even segments
            success = False
            if segment_method == '1':
                success = self.video_analyzer.analyze_engagement(video_path, self.transcript_manager.transcript, self.video_length)
            else:
                success = self.video_analyzer.create_even_segments(self.video_length)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to analyze video segments")
            
            # Generate shorts
            results = []
            for i, segment in enumerate(self.video_analyzer.engagement_data[:3], 1):
                print(f"\nGenerating short {i} from {segment['start']:.1f}s to {segment['end']:.1f}s")
                result = self.video_processor.generate_short(
                    video_path, segment['start'], segment['end'], i,
                    self.video_id, self.video_length, self.transcript_manager.transcript,
                    self.transcript_manager.language, self.face_detector
                )
                if result["url"]:
                    results.append({
                        "short_number": i,
                        "start_time": segment['start'],
                        "end_time": segment['end'],
                        "cloudinary_url": result["url"]
                    })
            
            # Clean up temporary video file
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    print(f"Cleaned up temporary video file: {video_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file {video_path}: {e}")
            
            return {
                "video_title": self.video_title,
                "video_id": self.video_id,
                "shorts_generated": results
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

# FastAPI endpoint
@app.post("/generate-shorts")
async def generate_shorts(request: VideoRequest):
    generator = YouTubeShortGenerator()
    return await generator.process_video(request.youtube_url, request.segment_method)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)