#!/usr/bin/env python3
"""
Web UI for the Code-Mixed Pedagogical Flow Extractor pipeline.

A lightweight Flask app that lets a user upload a video file and run the
full pipeline with real-time progress feedback via Server-Sent Events.

Usage:
    pip install flask
    python app.py
    # Open http://localhost:5000 in your browser
"""

import json
import os
import queue
import sys
import threading
import traceback
import uuid
from collections import defaultdict
from pathlib import Path
from time import sleep

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    Response,
)

# ─── paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Ensure required directories exist
for d in (
    DATA / "videos",
    DATA / "audios",
    DATA / "transcripts",
    DATA / "cleaned_transcripts",
    DATA / "concepts",
    DATA / "graphs",
    DATA / "analysis",
    DATA / "terminology_mappings",
    OUTPUTS,
):
    d.mkdir(parents=True, exist_ok=True)

# ─── Add project root to sys.path so pipeline imports work ───────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─── Lazy-loaded pipeline modules (heavy imports; load once on first use) ─
_pipeline_modules = {}


def _get_module(name: str):
    """Import a pipeline module lazily (avoids slow start-up)."""
    if name not in _pipeline_modules:
        import importlib
        _pipeline_modules[name] = importlib.import_module(f"pipeline.{name}")
    return _pipeline_modules[name]


app = Flask(__name__, template_folder="templates", static_folder="outputs")

# ─── in-memory job store ─────────────────────────────────────────────────
# job_id -> {"status": ..., "stem": ..., "events": Queue, "outputs": {...}}
_jobs: dict[str, dict] = {}


# ─── stage definitions ───────────────────────────────────────────────────

STAGES = [
    {
        "id": "download",
        "label": "Video Download",
        "description": "Downloading video from YouTube via yt-dlp",
    },
    {
        "id": "audio",
        "label": "Audio Extraction",
        "description": "Extracting mono 16 kHz WAV audio via ffmpeg",
    },
    {
        "id": "transcription",
        "label": "Transcription (Gemini)",
        "description": "Gemini-based code-mixed speech-to-text",
    },
    {
        "id": "standardisation",
        "label": "Linguistic Standardisation",
        "description": "Cleaning & normalising English text",
    },
    {
        "id": "terminology",
        "label": "Terminology Mapping",
        "description": "Mapping colloquial Indic terms to standard English academic terminology",
    },
    {
        "id": "concepts",
        "label": "Concept Extraction",
        "description": "spaCy + KeyBERT concept mining",
    },
    {
        "id": "graph",
        "label": "Graph Building (Gemini)",
        "description": "LLM-powered prerequisite graph construction",
    },
    {
        "id": "codemix",
        "label": "Code-Mix Analysis",
        "description": "CMI, script detection, switch-point analysis",
    },
    {
        "id": "visualisation",
        "label": "Visualisation",
        "description": "Interactive HTML, dependency diagram, canonical graph",
    },
    {
        "id": "roadmap",
        "label": "Learning Roadmap",
        "description": "Topologically sorted concept order for study planning",
    },
    {
        "id": "codemix_viz",
        "label": "Code-Mix Visualisation",
        "description": "CMI profile & language distribution chart",
    },
]


def _emit(job_id: str, event: str, data: dict):
    """Push an SSE event to the job's queue."""
    job = _jobs.get(job_id)
    if job:
        job["events"].put({"event": event, "data": data})


# ─── YouTube download helper ────────────────────────────────────────────

def _download_youtube(url: str) -> tuple[Path, str]:
    """Download a YouTube video and return (video_path, stem).

    Uses yt-dlp to fetch the best quality and convert to mp4.
    Returns the path of the downloaded file and its stem name.
    """
    import yt_dlp

    video_dir = DATA / "videos"
    info_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(info_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "video")

    # Sanitise the title for filesystem use (yt-dlp does this internally too)
    safe_title = yt_dlp.utils.sanitize_filename(title)

    dl_opts = {
        "format": "best",
        "outtmpl": str(video_dir / f"{safe_title}.%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file (should be .mp4 after conversion)
    video_path = video_dir / f"{safe_title}.mp4"
    if not video_path.exists():
        # Fallback: search for any file with this stem
        for p in video_dir.iterdir():
            if p.stem == safe_title:
                video_path = p
                break
    if not video_path.exists():
        raise RuntimeError(f"Download succeeded but file not found: {safe_title}")

    return video_path, safe_title


# ─── per-file pipeline helpers ───────────────────────────────────────────

def _stage_audio(stem: str):
    """Extract audio from video file."""
    mod = _get_module("audio_extractor")
    video_path = DATA / "videos" / f"{stem}.mp4"
    # Try common video extensions
    if not video_path.exists():
        for ext in (".mkv", ".webm", ".avi", ".mov", ".flv"):
            candidate = DATA / "videos" / (stem + ext)
            if candidate.exists():
                video_path = candidate
                break
    audio_path = DATA / "audios" / f"{stem}.wav"
    mod.extract_audio(str(video_path), str(audio_path))


def _stage_transcription(stem: str):
    """Transcribe audio via Gemini API."""
    mod = _get_module("speech_to_text")
    audio_path = DATA / "audios" / f"{stem}.wav"
    result = mod.transcribe_audio(str(audio_path))
    if result is None:
        raise RuntimeError("Transcription returned None – check GEMINI_API_KEY and audio file")
    # Save structured JSON
    json_out = DATA / "transcripts" / f"{stem}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def _stage_standardisation(stem: str):
    """Linguistically standardise the transcript."""
    mod = _get_module("linguistic_standardizer")
    transcript_json = DATA / "transcripts" / f"{stem}.json"
    with open(transcript_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", []) if isinstance(data, dict) else data
    detected_language = data.get("language", "unknown") if isinstance(data, dict) else "unknown"

    cleaned_segments = []
    nlp = mod.nlp  # spaCy model loaded by the module
    for seg in segments:
        english_text = seg.get("english_text", seg.get("text", ""))
        original_text = seg.get("original_text", english_text)
        lang = seg.get("language", detected_language)
        cleaned = mod.linguistic_standardize_segment(english_text)
        doc = nlp(cleaned)
        noun_count = sum(1 for t in doc if t.pos_ in ("NOUN", "PROPN"))
        if noun_count < 1:
            continue
        cleaned_segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "original_text": original_text,
            "english_text": english_text,
            "text": cleaned,
            "language": lang,
        })

    json_out = DATA / "cleaned_transcripts" / f"{stem}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_segments, f, indent=4, ensure_ascii=False)

def _stage_terminology(stem: str):
    """Map colloquial Indic terms to standard English academic terminology."""
    mod = _get_module("terminology_mapper")
    transcript_path = str(DATA / "transcripts" / f"{stem}.json")
    mod.generate_mapping(transcript_path)

def _stage_concepts(stem: str):
    """Extract and consolidate concepts."""
    mod = _get_module("concept_extract")

    # JSON with timestamps
    json_path = DATA / "cleaned_transcripts" / f"{stem}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    concept_occurrences: list[dict] = []
    for seg in segments:
        cleaned_text = seg.get("text", "")
        english_text = seg.get("english_text", cleaned_text)
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        extracted = set(mod.extract_concepts(cleaned_text))
        if english_text and english_text != cleaned_text:
            extracted.update(mod.extract_concepts(english_text))
        for c in extracted:
            concept_occurrences.append({"concept": c, "start": start, "end": end})

    normalized_occurrences = []
    for occ in concept_occurrences:
        normed = mod.normalize_concept(occ["concept"])
        if normed:
            normalized_occurrences.append({**occ, "normalized": normed})

    groups: dict[str, list[dict]] = defaultdict(list)
    for occ in normalized_occurrences:
        root = occ["normalized"].split()[-1]
        groups[root].append(occ)

    final_concepts_data: dict[str, list[dict]] = {}
    for _root, variants in groups.items():
        times = [{"start": v["start"], "end": v["end"]} for v in variants]
        best = min((v["normalized"] for v in variants), key=len)
        if best in final_concepts_data:
            final_concepts_data[best].extend(times)
        else:
            final_concepts_data[best] = times

    with open(DATA / "concepts" / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(final_concepts_data, f, indent=4, ensure_ascii=False)


def _stage_graph(stem: str):
    """Build the prerequisite graph via Gemini."""
    mod = _get_module("build_graph")
    concepts_file = str(DATA / "concepts" / f"{stem}.json")
    mod.generate_graph(concepts_file)
    # Verify output was created
    graph_file = DATA / "graphs" / f"{stem}.json"
    if not graph_file.exists():
        raise RuntimeError("Graph generation did not produce output – check Gemini API key / logs")


def _stage_codemix(stem: str):
    """Run code-mixing analysis on the transcript."""
    mod = _get_module("codemix_analyzer")
    transcript_path = str(DATA / "transcripts" / f"{stem}.json")
    result = mod.analyze_video(transcript_path)
    analysis_dir = DATA / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_out = analysis_dir / f"{stem}_codemix.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


def _stage_visualisation(stem: str):
    """Generate graph visualisations (interactive HTML, PNG, GEXF, etc.)."""
    mod = _get_module("visualize_graph")
    graph_json = str(DATA / "graphs" / f"{stem}.json")
    mod.process_graph(graph_json)


def _stage_roadmap(stem: str):
    """Generate a topologically sorted learning roadmap."""
    mod = _get_module("topological_sort")
    graph_json = str(DATA / "graphs" / f"{stem}.json")
    mod.generate_roadmap(graph_json)


def _stage_codemix_viz(stem: str):
    """Generate code-mix visualisations for this video."""
    mod = _get_module("visualize_codemix")
    analysis_file = DATA / "analysis" / f"{stem}_codemix.json"
    if not analysis_file.exists():
        return  # nothing to visualise
    with open(analysis_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    video_name = data.get("video", stem)
    out_dir = OUTPUTS / video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mod.make_codemix_profile(data, str(out_dir / "codemix_profile.png"))


# ─── collect downloadable artefacts ──────────────────────────────────────

# Maps stage_id -> list of (output_key, path) to check after that stage completes.
def _stage_outputs(stem: str, stage_id: str) -> dict:
    """Return outputs produced by a single stage (key -> abs path)."""
    out = {}

    def _add(key, path):
        if path.exists() and path.stat().st_size > 0:
            out[key] = str(path)

    if stage_id == "audio":
        _add("audio", DATA / "audios" / f"{stem}.wav")
    elif stage_id == "transcription":
        _add("transcript_json", DATA / "transcripts" / f"{stem}.json")
    elif stage_id == "standardisation":
        _add("cleaned_json", DATA / "cleaned_transcripts" / f"{stem}.json")
    elif stage_id == "terminology":
        _add("terminology_json", DATA / "terminology_mappings" / f"{stem}.json")
    elif stage_id == "concepts":
        _add("concepts_json", DATA / "concepts" / f"{stem}.json")
    elif stage_id == "graph":
        _add("graph_json", DATA / "graphs" / f"{stem}.json")
    elif stage_id == "codemix":
        _add("codemix_json", DATA / "analysis" / f"{stem}_codemix.json")
    elif stage_id == "visualisation":
        out_dir = OUTPUTS / stem
        _add("canonical_json", out_dir / "canonical.json")
        _add("gexf", out_dir / f"{stem}.gexf")
        _add("interactive_html", out_dir / "interactive.html")
        _add("dependency_png", out_dir / "dependency.png")
    elif stage_id == "roadmap":
        out_dir = OUTPUTS / stem
        _add("roadmap_json", out_dir / "learning_roadmap.json")
        _add("roadmap_png", out_dir / "learning_roadmap.png")
    elif stage_id == "codemix_viz":
        out_dir = OUTPUTS / stem
        _add("codemix_profile_png", out_dir / "codemix_profile.png")

    return out


# ─── background pipeline worker ─────────────────────────────────────────

def _pipeline_worker(job_id: str, stem: str, source: str = "file",
                     youtube_url: str | None = None):
    """Background thread that runs the full pipeline for one video.

    Args:
        source: "file" (video already saved to disk) or "youtube" (need to
                download first).
        youtube_url: Required when source == "youtube".
    """
    stage_funcs = [
        ("audio",           _stage_audio),
        ("transcription",   _stage_transcription),
        ("standardisation", _stage_standardisation),
        ("terminology",     _stage_terminology),
        ("concepts",        _stage_concepts),
        ("graph",           _stage_graph),
        ("codemix",         _stage_codemix),
        ("visualisation",   _stage_visualisation),
        ("roadmap",         _stage_roadmap),
        ("codemix_viz",     _stage_codemix_viz),
    ]
    try:
        # ── Optional: YouTube download stage ─────────────────────────────
        if source == "youtube":
            _emit(job_id, "stage", {"id": "download", "status": "running"})
            try:
                _video_path, stem = _download_youtube(youtube_url)
                _jobs[job_id]["stem"] = stem  # update with resolved title
                _emit(job_id, "stage", {"id": "download", "status": "done"})
            except Exception:
                tb = traceback.format_exc()
                _emit(job_id, "log", {"stage": "download", "text": tb[-2000:]})
                _emit(job_id, "stage", {"id": "download", "status": "error"})
                _jobs[job_id]["status"] = "error"
                _emit(job_id, "pipeline", {"status": "error", "stage": "download"})
                return
        else:
            # For file uploads, mark download stage as skipped/done instantly
            _emit(job_id, "stage", {"id": "download", "status": "done"})

        # ── Remaining stages ─────────────────────────────────────────────
        for stage_id, func in stage_funcs:
            _emit(job_id, "stage", {"id": stage_id, "status": "running"})
            try:
                func(stem)
                _emit(job_id, "stage", {"id": stage_id, "status": "done"})
                # Collect and emit outputs produced by this stage
                new_outputs = _stage_outputs(stem, stage_id)
                if new_outputs:
                    _jobs[job_id]["outputs"].update(new_outputs)
                    # Convert to download URLs for the frontend
                    urls = {k: f"/download/{job_id}/{k}" for k in new_outputs}
                    _emit(job_id, "outputs", {"files": urls})
            except Exception:
                tb = traceback.format_exc()
                _emit(job_id, "log", {"stage": stage_id, "text": tb[-2000:]})
                _emit(job_id, "stage", {"id": stage_id, "status": "error"})
                _jobs[job_id]["status"] = "error"
                _emit(job_id, "pipeline", {"status": "error", "stage": stage_id})
                return

        # All stages succeeded
        _jobs[job_id]["status"] = "done"
        _emit(job_id, "pipeline", {"status": "done"})

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _emit(job_id, "pipeline", {"status": "error", "message": str(exc)})


# ─── routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", stages=STAGES)


@app.route("/upload", methods=["POST"])
def upload():
    """Accept a video file, save it, and start the pipeline."""
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    stem = Path(f.filename).stem
    ext = Path(f.filename).suffix or ".mp4"
    video_path = DATA / "videos" / (stem + ext)
    f.save(str(video_path))

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "stem": stem,
        "events": queue.Queue(),
        "outputs": {},
    }

    t = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, stem),
        kwargs={"source": "file"},
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "stem": stem})


@app.route("/start_url", methods=["POST"])
def start_url():
    """Accept a YouTube URL and start the pipeline (download + process)."""
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "stem": "downloading…",
        "events": queue.Queue(),
        "outputs": {},
    }

    t = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, ""),
        kwargs={"source": "youtube", "youtube_url": url},
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id, "stem": "(downloading)"})


@app.route("/events/<job_id>")
def events(job_id):
    """Server-Sent Events stream for real-time progress."""
    job = _jobs.get(job_id)
    if not job:
        return "Job not found", 404

    def generate():
        while True:
            try:
                msg = job["events"].get(timeout=30)
                yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
                # If pipeline finished, send one last keep-alive then break
                if msg["event"] == "pipeline":
                    yield f"event: done\ndata: {{}}\n\n"
                    break
            except queue.Empty:
                # Send keep-alive comment to prevent timeout
                yield ": heartbeat\n\n"
                if job["status"] in ("done", "error"):
                    break

    return Response(generate(), mimetype="text/event-stream")


@app.route("/download/<job_id>/<key>")
def download_file(job_id, key):
    """Serve a specific output file."""
    job = _jobs.get(job_id)
    if not job:
        return "Job not found", 404
    abspath = job.get("outputs", {}).get(key)
    if not abspath or not Path(abspath).exists():
        return "File not found", 404
    p = Path(abspath)
    return send_from_directory(str(p.parent), p.name, as_attachment=(key != "interactive_html"))


@app.route("/view/<job_id>/interactive")
def view_interactive(job_id):
    """Serve the interactive HTML visualisation inline."""
    job = _jobs.get(job_id)
    if not job:
        return "Job not found", 404
    abspath = job.get("outputs", {}).get("interactive_html")
    if not abspath or not Path(abspath).exists():
        return "Interactive visualisation not available yet", 404
    return send_from_directory(str(Path(abspath).parent), Path(abspath).name)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Code-Mixed Pedagogical Flow Extractor – Web UI    ║")
    print("║   Open http://localhost:5000 in your browser        ║")
    print("╚══════════════════════════════════════════════════════╝")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
