import os
import logging
from typing import List, Dict, Any
import torch
from moviepy.editor import VideoFileClip, CompositeVideoClip, ColorClip, ImageClip
import whisper
import srt
from datetime import timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import tiktoken
import textwrap
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Paths
    FONT_PATH = os.getenv('FONT_PATH', "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    JAPANESE_FONT_PATH = os.getenv('JAPANESE_FONT_PATH', "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    TEMP_AUDIO_FILE = os.getenv('TEMP_AUDIO_FILE', "temp_audio.wav")

    # Video processing
    DEFAULT_SUBTITLE_HEIGHT = int(os.getenv('DEFAULT_SUBTITLE_HEIGHT', 200))
    DEFAULT_FONT_SIZE = int(os.getenv('DEFAULT_FONT_SIZE', 32))
    MAX_SUBTITLE_LINES = int(os.getenv('MAX_SUBTITLE_LINES', 3))

    # Video encoding
    VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'libx264')
    AUDIO_CODEC = os.getenv('AUDIO_CODEC', 'aac')
    VIDEO_PRESET = os.getenv('VIDEO_PRESET', 'medium')
    CRF = os.getenv('CRF', '23')
    PIXEL_FORMAT = os.getenv('PIXEL_FORMAT', 'yuv420p')

    # Tiktoken related settings
    TIKTOKEN_MODEL = "cl100k_base"
    MAX_TOKENS_PER_CHUNK = 4000

    # OpenAI settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    DEFAULT_GPT_MODEL = "gpt-4o"
    GPT_MAX_TOKENS = 4000

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubtitleProcessor:
    def __init__(self, video_path: str, srt_path: str):
        self.video_path = video_path
        self.srt_path = srt_path
        self.temp_files = []

    def cleanup_temp_files(self):
        logger.info("Cleaning up temporary files...")
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")

class SRTTranslator:
    def __init__(self, model: str = Config.DEFAULT_GPT_MODEL, temperature: float = 0.3):
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key is required. Set it in the environment variable 'OPENAI_API_KEY'.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def translate_srt(self, input_srt: str, output_srt: str, source_lang: str, target_lang: str):
        with open(input_srt, 'r', encoding='utf-8') as f:
            subtitle_generator = srt.parse(f.read())
            subtitles = list(subtitle_generator)

        translated_subtitles = []
        for subtitle in tqdm(subtitles, desc="Translating subtitles"):
            translated_content = self.translate_text(subtitle.content, source_lang, target_lang)
            translated_subtitle = srt.Subtitle(
                index=subtitle.index,
                start=subtitle.start,
                end=subtitle.end,
                content=translated_content
            )
            translated_subtitles.append(translated_subtitle)

        with open(output_srt, 'w', encoding='utf-8') as f:
            f.write(srt.compose(translated_subtitles))

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text from {source_lang} to {target_lang}. Maintain the original meaning and nuance as much as possible. Do not modify any formatting or line breaks."},
                {"role": "user", "content": text}
            ],
            temperature=self.temperature,
            max_tokens=Config.GPT_MAX_TOKENS
        )
        return response.choices[0].message.content.strip()

class SRTGenerator(SubtitleProcessor):
    def __init__(self, video_path: str, output_srt: str, model_name: str, language: str = "japanese", translate: bool = False, api_key: str = None):
        super().__init__(video_path, output_srt)
        self.model_name = model_name
        self.translate = translate
        self.tokenizer = tiktoken.get_encoding(Config.TIKTOKEN_MODEL)
        self.language = language
        self.api_key = api_key

    def run(self):
        try:
            self.extract_audio()
            transcription = self.transcribe_audio()
            chunks = self.split_into_chunks(transcription)
            results = self.process_chunks(chunks)
            self.create_srt(results)

            logger.info(f"SRT file has been generated: {self.srt_path}")
        finally:
            self.cleanup_temp_files()

    def extract_audio(self):
        logger.info("Extracting audio from video...")
        video = VideoFileClip(self.video_path)
        video.audio.write_audiofile(Config.TEMP_AUDIO_FILE)
        self.temp_files.append(Config.TEMP_AUDIO_FILE)

    def transcribe_audio(self) -> Dict[str, Any]:
        logger.info("Transcribing audio with Whisper...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        logger.info(f"Loading Whisper model: {self.model_name}")
        model = whisper.load_model(self.model_name).to(device)

        logger.info(f"Performing task: transcribe with language: {self.language}")
        result = model.transcribe(Config.TEMP_AUDIO_FILE, task="transcribe", language=self.language)
        return result

    def split_into_chunks(self, transcription: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("Splitting transcription into chunks...")
        chunks = []
        current_chunk = {"text": "", "segments": []}
        current_tokens = 0

        for segment in transcription['segments']:
            segment_tokens = self.tokenizer.encode(segment['text'])
            if current_tokens + len(segment_tokens) > Config.MAX_TOKENS_PER_CHUNK:
                chunks.append(current_chunk)
                current_chunk = {"text": "", "segments": []}
                current_tokens = 0
            
            current_chunk['text'] += segment['text'] + " "
            current_chunk['segments'].append(segment)
            current_tokens += len(segment_tokens)

        if current_chunk['segments']:
            chunks.append(current_chunk)

        return chunks

    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info("Processing chunks...")
        results = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            results.extend(chunk['segments'])
        return results

    def create_srt(self, results: List[Dict[str, Any]]):
        logger.info("Creating SRT file...")
        subs = []
        for i, segment in enumerate(results, start=1):
            start = timedelta(seconds=segment['start'])
            end = timedelta(seconds=segment['end'])
            text = segment['text']
            sub = srt.Subtitle(index=i, start=start, end=end, content=text)
            subs.append(sub)
        
        with open(self.srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subs))

class SubtitleAdder(SubtitleProcessor):
    def __init__(self, video_path: str, output_video: str, input_srt: str, subtitle_height: int = Config.DEFAULT_SUBTITLE_HEIGHT):
        super().__init__(video_path, input_srt)
        self.output_video = output_video
        self.subtitle_height = subtitle_height

    def run(self):
        try:
            subs = self.load_srt(self.srt_path)
            self.add_subtitles_to_video(subs)
            logger.info(f"Video with subtitles has been generated: {self.output_video}")
        finally:
            self.cleanup_temp_files()

    @staticmethod
    def load_srt(srt_path: str) -> List[srt.Subtitle]:
        logger.info(f"Loading SRT file: {srt_path}")
        with open(srt_path, 'r', encoding='utf-8') as f:
            return list(srt.parse(f.read()))

    def add_subtitles_to_video(self, subs: List[srt.Subtitle]):
        logger.info(f"Adding subtitles to video with subtitle space height of {self.subtitle_height} pixels...")
        video = VideoFileClip(self.video_path)
        
        original_width, original_height = video.w, video.h
        new_height = original_height + self.subtitle_height
        
        background = ColorClip(size=(original_width, new_height), color=(0,0,0), duration=video.duration)
        video_clip = video.set_position((0, 0))
        
        subtitle_clips = [
            self.create_subtitle_clip(sub.content, original_width)
                .set_start(sub.start.total_seconds())
                .set_end(sub.end.total_seconds())
                .set_position((0, original_height))
            for sub in subs
        ]
        
        final_video = CompositeVideoClip([background, video_clip] + subtitle_clips, size=(original_width, new_height))
        final_video = final_video.set_duration(video.duration)
        
        final_video.write_videofile(
            self.output_video, 
            codec=Config.VIDEO_CODEC, 
            audio_codec=Config.AUDIO_CODEC,
            preset=Config.VIDEO_PRESET,
            ffmpeg_params=['-crf', Config.CRF, '-pix_fmt', Config.PIXEL_FORMAT],
            verbose=False,
            logger=None
        )

    @staticmethod
    def create_subtitle_clip(txt: str, video_width: int, font_size: int = Config.DEFAULT_FONT_SIZE, max_lines: int = Config.MAX_SUBTITLE_LINES) -> ImageClip:
        if any(ord(char) > 127 for char in txt):
            font_path = Config.JAPANESE_FONT_PATH
        else:
            font_path = Config.FONT_PATH

        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            logger.warning(f"Failed to load font from {font_path}. Falling back to default font.")
            font = ImageFont.load_default()
        
        max_char_count = int(video_width / (font_size * 0.6))
        wrapped_text = textwrap.fill(txt, width=max_char_count)
        lines = wrapped_text.split('\n')[:max_lines]
        
        dummy_img = Image.new('RGB', (video_width, font_size * len(lines)))
        dummy_draw = ImageDraw.Draw(dummy_img)
        max_line_width = max(dummy_draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        total_height = sum(dummy_draw.textbbox((0, 0), line, font=font)[3] for line in lines)
        
        img_width, img_height = video_width, total_height + 20
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        y_text = 10
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x_text = (img_width - bbox[2]) // 2
            
            for adj in range(-2, 3):
                for adj2 in range(-2, 3):
                    draw.text((x_text+adj, y_text+adj2), line, font=font, fill=(0, 0, 0, 255))
            
            draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255, 255))
            y_text += bbox[3]
        
        return ImageClip(np.array(img))

def main():
    parser = argparse.ArgumentParser(description="Subtitle Generator and Adder", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="action", required=True)

    # Common arguments for generate and add commands
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--input", required=True, help="Input video file path")

    # Generate subparser for creating subtitles
    generate_parser = subparsers.add_parser("generate", parents=[common_parser])
    generate_parser.add_argument("--output_srt", required=True, help="Output SRT file path")
    generate_parser.add_argument("--model", default="large-v3", help="Whisper model name (default: large-v3)")
    generate_parser.add_argument("--language", default="Japanese", help="Language of the audio (default: Japanese)")
    generate_parser.add_argument("--translate", action="store_true", help="Translate the audio to English")

    # Add subparser for adding subtitles to a video
    add_parser = subparsers.add_parser("add", parents=[common_parser])
    add_parser.add_argument("--output_video", required=True, help="Output video file path")
    add_parser.add_argument("--input_srt", required=True, help="Input SRT file path")

    # Translate subparser for translating an SRT file
    translate_parser = subparsers.add_parser("translate")
    translate_parser.add_argument("--input_srt", required=True, help="Input SRT file path")
    translate_parser.add_argument("--output_srt", required=True, help="Output SRT file path")
    translate_parser.add_argument("--source_lang", default="Japanese", help="Source language of the SRT file (default: Japanese)")
    translate_parser.add_argument("--target_lang", default="English", help="Target language for translation (default: English)")
    translate_parser.add_argument("--temperature", type=float, default=0.3, help="Temperature setting for OpenAI (default: 0.3)")

    args = parser.parse_args()

    if args.action == "generate":
        generator = SRTGenerator(args.input, args.output_srt, args.model, args.language, args.translate)
        generator.run()
    elif args.action == "add":
        adder = SubtitleAdder(args.input, args.output_video, args.input_srt)
        adder.run()
    elif args.action == "translate":
        translator = SRTTranslator(temperature=args.temperature)
        translator.translate_srt(args.input_srt, args.output_srt, args.source_lang, args.target_lang)
        logger.info(f"Translation completed. Output saved to {args.output_srt}")

if __name__ == "__main__":
    main()
