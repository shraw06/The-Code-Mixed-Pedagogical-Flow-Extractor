#!/usr/bin/env python3
"""
Step 3 – Speech-to-Text via Gemini API (transcription-first approach).

Instead of Whisper's translate mode (which directly translated code-mixed
audio into English, losing the original phrasing), this module:

  1. Sends each audio file to Gemini for **transcription** — preserving the
     original code-mixed speech (Hinglish, Tamil-English, Telugu-English,
     Marathi-English, etc.).
  2. Asks Gemini to return timestamped segments with:
       - original_text  : the verbatim code-mixed transcription
       - english_text   : a faithful English translation of that segment
       - language        : detected language/code-mixing label
  3. Writes structured JSON into data/transcripts/.

This preserves the raw code-mixed phrasing for downstream linguistic
standardisation and concept extraction while also providing an English
translation for LLM-based analysis.

Usage:
    python pipeline/speech_to_text.py
"""

import json
import glob
import os
import re
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# ── Directories ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audios"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_ascii_name(name: str) -> str:
    """Return an ASCII-only filename safe for HTTP headers.

    Converts accented / fullwidth characters to their ASCII closest equivalent
    via NFKD normalisation, then strips anything still non-ASCII.
    """
    # NFKD decomposition collapses fullwidth chars (｜→|) and accented letters
    normalized = unicodedata.normalize("NFKD", name)
    # Encode to ASCII dropping non-encodable bytes
    ascii_name = normalized.encode("ascii", errors="ignore").decode("ascii")
    # Replace characters that are problematic in filenames / headers
    ascii_name = re.sub(r"[^\w\s.\-]", "_", ascii_name)
    ascii_name = re.sub(r"\s+", "_", ascii_name).strip("_")
    return ascii_name or "audio"


def _repair_json_text(text: str) -> str:
    """Apply heuristic repairs to malformed JSON produced by Gemini.

    Known issues this handles:
      - Double-decimal numbers: "8.56.0" → "856.0"  (keep last dot only)
      - Smart / curly quotes
      - Trailing commas before } or ]
      - Stray non-JSON text between array/object elements
    """
    # Fix double-decimal numbers like  8.56.0  →  856.0
    # Pattern: a number with two or more dots, e.g.  123.45.67  or  8.56.0
    # Strategy: keep only the LAST dot as the decimal point.
    def _fix_multi_dot_number(m):
        raw = m.group(0)
        parts = raw.split(".")
        # join all parts except last with no separator, then dot + last part
        return "".join(parts[:-1]) + "." + parts[-1]

    text = re.sub(r"\b\d+(?:\.\d+){2,}\b", _fix_multi_dot_number, text)

    # Normalise smart / curly quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*(?=[}\]])", "", text)

    # Fix missing closing brace between array elements.
    # Gemini sometimes emits:
    #     "language": "Marathi-English"
    #     ,
    #     {
    # instead of:
    #     "language": "Marathi-English"
    #     },
    #     {
    # Detect a string/number/bool value followed by optional whitespace,
    # a comma, then a `{` on the next non-blank line (i.e. start of the
    # next object in the array) — and insert the missing  }  before the comma.
    text = re.sub(
        r'("(?:[^"\\]|\\.)*"|true|false|null|\d+(?:\.\d+)?)'  # value token
        r'(\s*)\n(\s*),(\s*)\n(\s*)\{',                        # \n  ,\n  {
        r'\1\2\n\3},\4\n\5{',
        text,
    )

    # Remove stray lines between JSON object elements.
    # Heuristic: a line that doesn't start with whitespace + a JSON token
    # and sits between  }  and  {  or  }  and  ]  is probably hallucinated text.
    # We remove lines that don't look like JSON structure.
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep blank lines, and lines that start with JSON structural chars or string keys
        if (
            not stripped
            or stripped[0] in '{[]},"'
            or re.match(r'^[\s]*"', line)
            or re.match(r"^[\s]*[\d\-]", line)        # number values
            or re.match(r"^[\s]*(true|false|null)", line)
        ):
            cleaned_lines.append(line)
        # else: skip the stray hallucinated line
    text = "\n".join(cleaned_lines)

    return text


def _extract_json(text: str):
    """Robustly extract JSON from an LLM response that may contain markdown
    fences, commentary, etc."""
    # strip ALL markdown code fences (opening and closing)
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try a cascade: raw → repaired → substring extraction

    for t in (text, _repair_json_text(text)):
        # direct parse
        try:
            return json.loads(t)
        except Exception:
            pass

        # outermost { ... }
        fb = t.find("{")
        lb = t.rfind("}")
        if fb != -1 and lb > fb:
            candidate = t[fb : lb + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # outermost [ ... ]
        fb = t.find("[")
        lb = t.rfind("]")
        if fb != -1 and lb > fb:
            candidate = t[fb : lb + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

    # ── Last resort: iterative missing-brace insertion ───────────────────
    # Walk through the repaired text and try inserting `}` at every position
    # where json.loads reports "Expecting property name" (symptom of a
    # missing closing brace in an array of objects).
    repaired = _repair_json_text(text)
    fb = repaired.find("{")
    lb = repaired.rfind("}")
    if fb != -1 and lb > fb:
        candidate = repaired[fb : lb + 1]
        for _ in range(20):  # at most 20 repair rounds
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                if "Expecting property name" in e.msg or "Expecting ',' delimiter" in e.msg:
                    # Insert a } just before the error position
                    candidate = candidate[: e.pos] + "}" + candidate[e.pos :]
                else:
                    break

    return None


# ── Core transcription prompt ────────────────────────────────────────────

TRANSCRIBE_PROMPT = """\
You are an expert multilingual transcription system.

I am providing you an audio file from an Indian educational video lecture.
The speaker uses heavy **code-mixing** — they switch freely between an
Indian language (Hindi, Tamil, Telugu, Marathi, etc.) and English technical
terms.

Your task:
1. **Transcribe** the audio faithfully — preserve the original code-mixed
   speech exactly as spoken (use romanised / transliterated form for the
   Indic portions so downstream NLP tools can process it).
   Do NOT translate at this stage — keep the original words.
2. Split the transcript into segments of roughly 10-30 seconds each.
   For each segment provide:
     - "start"          : float, start time in seconds
     - "end"            : float, end time in seconds
     - "original_text"  : the verbatim code-mixed transcription
     - "english_text"   : a faithful, natural English translation of just
                          that segment (translate the Indic portions to
                          English while keeping English technical terms
                          unchanged)
     - "language"       : a short label for the detected language mix,
                          e.g. "Hindi-English", "Tamil-English",
                          "Telugu-English", "Marathi-English", etc.
3. Return ONLY a JSON object with these keys:
     {
       "language": "<primary code-mix label>",
       "segments": [ { "start", "end", "original_text", "english_text", "language" }, ... ]
     }
   No markdown fences, no commentary — just the JSON.
"""


# ── Core transcription function ──────────────────────────────────────────

def transcribe_audio(audio_path: str, api_key: str | None = None) -> dict | None:
    """Transcribe a single audio file via Gemini and return structured dict."""

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"  ✗ GEMINI_API_KEY not set — skipping {audio_path}")
        return None

    if genai is None:
        print("  ✗ google-genai package not installed.  pip install google-genai")
        return None

    client = genai.Client(api_key=api_key)

    # ── Upload the audio file ─────────────────────────────────────────────
    # httpx (used internally by google-genai) encodes the filename in an
    # ASCII HTTP header, so any non-ASCII characters (fullwidth ｜, Devanagari,
    # etc.) in the original filename will raise UnicodeEncodeError.
    # Fix: copy the file to a temp directory under an ASCII-safe name first.
    original_path = Path(audio_path)
    safe_stem = _safe_ascii_name(original_path.stem)
    suffix = original_path.suffix  # e.g. ".wav"

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, safe_stem + suffix)
    try:
        shutil.copy2(str(original_path), tmp_path)
        print(f"  ↑ Uploading {original_path.name} (as {safe_stem + suffix}) …")
        uploaded = client.files.upload(file=tmp_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Wait until the file is ACTIVE (processed by Gemini)
    max_wait = 120  # seconds
    waited = 0
    while uploaded.state.name != "ACTIVE":
        if waited >= max_wait:
            print(f"  ✗ File processing timed out after {max_wait}s")
            return None
        time.sleep(5)
        waited += 5
        uploaded = client.files.get(name=uploaded.name)

    print(f"  ✓ File ready ({waited}s).  Sending transcription request …")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai_types.Content(
                parts=[
                    genai_types.Part.from_uri(
                        file_uri=uploaded.uri,
                        mime_type=uploaded.mime_type or "audio/wav",
                    ),
                    genai_types.Part.from_text(text=TRANSCRIBE_PROMPT),
                ]
            )
        ],
        config=genai_types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )

    # Extract text from response
    raw = ""
    if hasattr(response, "text") and response.text:
        raw = response.text
    elif hasattr(response, "content"):
        raw = (
            response.content.decode("utf-8", errors="ignore")
            if isinstance(response.content, (bytes, bytearray))
            else str(response.content)
        )
    else:
        raw = str(response)

    # Save raw response for debugging
    debug_path = TRANSCRIPT_DIR / (Path(audio_path).stem + ".raw_debug.txt")
    debug_path.write_text(raw, encoding="utf-8")

    # Parse JSON
    data = _extract_json(raw)
    if data is None:
        print(f"  ✗ Could not parse JSON from Gemini response.")
        print(f"    Raw response saved → {debug_path}")
        return None

    return _normalise_result(data)


def _normalise_result(data: dict | list) -> dict:
    """Normalise a parsed Gemini response into the canonical transcript format."""
    if isinstance(data, list):
        # Gemini returned a bare segments array
        data = {"language": "unknown", "segments": data}
    elif isinstance(data, dict) and "segments" not in data:
        data = {"language": data.get("language", "unknown"), "segments": []}

    # Ensure every segment has the required fields
    for seg in data.get("segments", []):
        # Coerce start/end to float (handles stringified numbers)
        for key in ("start", "end"):
            try:
                seg[key] = float(seg.get(key, 0.0))
            except (TypeError, ValueError):
                seg[key] = 0.0
        seg.setdefault("original_text", seg.get("text", ""))
        seg.setdefault("english_text", seg.get("translation", seg.get("original_text", "")))
        seg.setdefault("language", data.get("language", "unknown"))
        # Backward-compat "text" key used by linguistic_standardizer & concept_extract
        seg["text"] = seg["english_text"]

    return data


# ── Main ─────────────────────────────────────────────────────────────────

def process_all():
    audio_files = sorted(glob.glob(str(AUDIO_DIR / "*.wav")))

    # Also discover transcripts that need re-parsing from raw_debug files
    # (e.g. when audio files have been removed but cached responses exist)
    raw_debug_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.raw_debug.txt")))
    all_stems = set()
    stem_to_audio = {}
    for af in audio_files:
        s = Path(af).stem
        all_stems.add(s)
        stem_to_audio[s] = af
    for rf in raw_debug_files:
        s = Path(rf).stem  # e.g. "Foo.raw_debug" — need to strip ".raw_debug"
        # The file is named <stem>.raw_debug.txt, Path().stem gives "<stem>.raw_debug"
        actual_stem = Path(rf).name.replace(".raw_debug.txt", "")
        all_stems.add(actual_stem)

    if not all_stems:
        print("No .wav files in", AUDIO_DIR, "and no raw_debug files in", TRANSCRIPT_DIR)
        return

    api_key = os.environ.get("GEMINI_API_KEY")

    for stem in sorted(all_stems):
        json_out = TRANSCRIPT_DIR / f"{stem}.json"
        debug_file = TRANSCRIPT_DIR / f"{stem}.raw_debug.txt"
        audio = stem_to_audio.get(stem)

        # ── Skip files that already have a valid transcript with segments ──
        if json_out.exists():
            try:
                existing = json.loads(json_out.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and len(existing.get("segments", [])) > 0:
                    print(f"\n── Skipping (already has {len(existing['segments'])} segments): {stem}")
                    continue
            except Exception:
                pass

        print(f"\n── Transcribing: {stem}")

        # ── Try to re-parse an existing raw_debug file first (saves API calls) ──
        result = None
        if debug_file.exists():
            print(f"  ↻ Found existing raw response, attempting re-parse …")
            raw = debug_file.read_text(encoding="utf-8")
            parsed = _extract_json(raw)
            if parsed is not None:
                result = _normalise_result(parsed)
                if result and len(result.get("segments", [])) > 0:
                    print(f"  ✓ Re-parsed {len(result['segments'])} segments from cached raw response")
                else:
                    result = None

        # ── If re-parse failed, call Gemini API ──
        if result is None:
            if not audio:
                print(f"  ✗ No audio file and raw re-parse failed — skipping {stem}")
                continue
            if not api_key:
                print(f"  ✗ GEMINI_API_KEY not set — skipping {stem}")
                continue
            result = transcribe_audio(audio, api_key)
        if result is None:
            continue

        # ── Save structured JSON ──
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"  ✓ JSON transcript → {json_out}")

        # Rate-limit between files
        time.sleep(5)


if __name__ == "__main__":
    process_all()