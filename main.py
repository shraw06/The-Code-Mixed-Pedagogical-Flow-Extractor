#!/usr/bin/env python3
"""
Orchestrator for the Code-Mixed Pedagogical Flow Extractor pipeline.

This script runs the full pipeline in sequence:
  1. video_downloader       — download educational videos via yt-dlp
  2. audio_extractor        — extract audio tracks (WAV) via ffmpeg
  3. speech_to_text         — Gemini-based transcription preserving code-mixed speech
  4. linguistic_standardizer— clean & normalise English text, preserve originals
  5. concept_extract        — spaCy + KeyBERT concept extraction with timestamps
  6. build_graph            — LLM-powered pedagogical prerequisite graph (DAG)
  7. codemix_analyzer       — quantitative code-mixing analysis (CMI, switches)
  8. visualize_graph        — per-video dependency diagrams, interactive HTML
  9. topological_sort       — learning roadmap (topologically sorted concept order + flowchart)
 10. visualize_codemix      — code-mix profiles, cross-video comparison

Each step is executed as a separate Python process to avoid importing
modules that execute work at import time.

Usage:
    python main.py

You can also import the functions from other code:
    from main import run_all, download_videos
"""

from pathlib import Path
import subprocess
import sys
import shutil
import os


ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "pipeline"


def _run_script(script_name):
    """Run a pipeline script (relative to the pipeline directory).

    Returns True on success, False otherwise. Prints stdout/stderr to console.
    """
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        print(f"Missing script: {script_path}")
        return False

    print(f"\n--- Running: {script_name} ---")
    cmd = [sys.executable, str(script_path)]
    try:
        completed = subprocess.run(cmd, check=True)
        print(f"{script_name} finished (rc={completed.returncode})")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with returncode {e.returncode}")
        return False


def download_videos():
    return _run_script("video_downloader.py")


def extract_audio():
    return _run_script("audio_extractor.py")


def speech_to_text():
    return _run_script("speech_to_text.py")


def linguistic_standardizer():
    return _run_script("linguistic_standardizer.py")


def concept_extract():
    return _run_script("concept_extract.py")


def build_graphs():
    return _run_script("build_graph.py")

def codemix_analyze():
    return _run_script("codemix_analyzer.py")

def visualize_graphs():
    return _run_script("visualize_graph.py")

def topological_sort():
    return _run_script("topological_sort.py")

def visualize_codemix():
    return _run_script("visualize_codemix.py")


def run_all():
    steps = [
        ("download_videos", download_videos),
        ("extract_audio", extract_audio),
        ("speech_to_text", speech_to_text),
        ("linguistic_standardizer", linguistic_standardizer),
        ("concept_extract", concept_extract),
        ("build_graph", build_graphs),
        ("codemix_analyzer", codemix_analyze),
        ("visualize_graphs", visualize_graphs),
        ("topological_sort", topological_sort),
        ("visualize_codemix", visualize_codemix),
    ]

    for name, fn in steps:
        ok = fn()
        if not ok:
            print(f"Pipeline stopped: step '{name}' failed.")
            return False

    print("\nPipeline completed successfully.")
    return True


if __name__ == "__main__":
    run_all()
