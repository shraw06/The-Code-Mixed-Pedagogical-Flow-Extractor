#!/usr/bin/env python3
"""
Step 2 – Extract audio (mono 16kHz PCM WAV) from downloaded videos.
"""

import subprocess
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
AUDIO_DIR = ROOT / "data" / "audios"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path, audio_path):
    command = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    for video in sorted(glob.glob(str(VIDEO_DIR / "*.mp4"))):
        stem = Path(video).stem
        audio_file = AUDIO_DIR / f"{stem}.wav"
        if audio_file.exists():
            print(f"  ⏭ Already exists: {audio_file.name}")
            continue
        print(f"  Extracting audio: {stem}")
        extract_audio(video, audio_file)
        print(f"  ✓ {audio_file}")