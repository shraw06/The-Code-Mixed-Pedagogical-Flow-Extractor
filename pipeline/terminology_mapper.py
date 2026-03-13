#!/usr/bin/env python3
"""
Step 4c – Terminology Mapping (colloquial Indic → standard English).

Produces an explicit, traceable mapping of colloquial Indic terms and
code-mixed analogies back to their standard English academic terminology.

For each transcript segment that contains code-mixed content, this module
sends the ``original_text`` and ``english_text`` to Gemini and asks for a
phrase-level alignment:

  • Which Indic colloquial phrase in the original maps to which English
    academic term in the translation?
  • Is the term technical (academic) or conversational?
  • What is the standard English academic form?

Reads  : data/transcripts/<stem>.json
Writes : data/terminology_mappings/<stem>.json
         outputs/<stem>/terminology_mapping.json  (copy for output bundle)

Usage:
    python pipeline/terminology_mapper.py               # process all
    python pipeline/terminology_mapper.py <transcript>   # process one
"""

import glob
import json
import os
import re
import sys
from pathlib import Path
from time import sleep

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# ── Directories ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
MAPPING_DIR = ROOT / "data" / "terminology_mappings"
OUTPUT_DIR = ROOT / "outputs"
MAPPING_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt ───────────────────────────────────────────────────────────────

_MAPPING_PROMPT = """\
You are a linguistics expert specialising in Indian code-mixed educational speech.

I will give you a batch of transcript segments from a code-mixed lecture.
Each segment has:
  - "original_text": the verbatim code-mixed transcription
  - "english_text": a faithful English translation of that segment
  - "language": the detected language mix label
  - "start" / "end": timestamps in seconds

YOUR TASK — for EACH segment produce a list of **terminology mappings**:

For every Indic colloquial term, phrase, or analogy in the original_text
that corresponds to a standard English academic or technical concept,
produce a mapping entry.

Rules:
1. Only map phrases where the Indic portion carries **meaningful content**
   (skip pure filler words like "bhaiya", "dosto", "aaj", "dekho" unless
   they are part of a larger analogy).
2. Focus on:
   - Technical terms spoken in the regional language
     (e.g. "diye gaye matrix" → "Given Matrix")
   - Code-mixed analogies where a colloquial phrase explains a concept
     (e.g. "chain jodna" → "Linked List Traversal")
   - Domain-specific colloquialisms
     (e.g. "sawal" → "Problem", "nikal lo" in math context → "Compute")
3. Provide the **standard English academic term** — the formal name used
   in textbooks, not just a literal translation.
4. Mark each mapping as "technical" or "conversational".
5. If a segment has NO meaningful Indic terms to map (e.g. it is entirely
   in English), return an empty array for that segment.

OUTPUT FORMAT — return ONLY a JSON object (no markdown, no commentary):
{
  "mappings": [
    {
      "segment_index": 0,
      "start": 0.0,
      "end": 10.5,
      "entries": [
        {
          "original_phrase": "diye gaye matrix",
          "language": "Hindi",
          "literal_translation": "given matrix",
          "standard_term": "Given Matrix",
          "category": "technical",
          "context": "...short surrounding context from original_text..."
        }
      ]
    }
  ]
}
"""

# ── Helpers ──────────────────────────────────────────────────────────────

def _extract_json(text: str):
    """Robustly parse JSON from an LLM response."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    for t in (text,):
        try:
            return json.loads(t)
        except Exception:
            pass
        # Try outermost { … }
        fb = t.find("{")
        lb = t.rfind("}")
        if fb != -1 and lb > fb:
            try:
                return json.loads(t[fb : lb + 1])
            except Exception:
                pass
        # Try outermost [ … ]
        fb = t.find("[")
        lb = t.rfind("]")
        if fb != -1 and lb > fb:
            try:
                return json.loads(t[fb : lb + 1])
            except Exception:
                pass
    return None


def _is_code_mixed(seg: dict) -> bool:
    """Return True if the segment likely contains Indic content worth mapping."""
    orig = seg.get("original_text", "")
    eng = seg.get("english_text", "")
    lang = seg.get("language", "")

    # If original and english are identical, no code-mixing to map
    if orig.strip().lower() == eng.strip().lower():
        return False

    # If the language label says pure English, skip
    if lang.lower() in ("english", "en", ""):
        return False

    return True


def _existing_mapping_is_valid(stem: str) -> bool:
    """Return True if a terminology mapping already exists with real content."""
    mapping_file = MAPPING_DIR / f"{stem}.json"
    if not mapping_file.exists():
        return False
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Must be a dict with a non-empty "mappings" list
        if not isinstance(data, dict):
            return False
        mappings = data.get("mappings", [])
        if not isinstance(mappings, list) or len(mappings) == 0:
            return False
        # At least one segment must have at least one entry
        return any(
            isinstance(m, dict) and len(m.get("entries", [])) > 0
            for m in mappings
        )
    except (json.JSONDecodeError, OSError):
        return False


# ── Core mapping function ────────────────────────────────────────────────

def generate_mapping(transcript_path: str, api_key: str | None = None,
                     force: bool = False) -> dict:
    """Generate terminology mappings for a single transcript file.

    Args:
        transcript_path: Path to a transcript JSON (data/transcripts/<stem>.json).
        api_key: Optional Gemini API key; falls back to env var.
        force: If True, regenerate even if a valid mapping already exists.

    Returns:
        A dict with "video", "language", "total_segments",
        "mapped_segments", "total_mappings", and "mappings" keys.
    """
    stem = Path(transcript_path).stem

    # Skip if a valid mapping already exists (unless forced)
    if not force and _existing_mapping_is_valid(stem):
        print(f"  ✓ Terminology mapping already exists for {stem} — skipping.")
        with open(MAPPING_DIR / f"{stem}.json", "r", encoding="utf-8") as f:
            return json.load(f)

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"  ✗ GEMINI_API_KEY not set — skipping {transcript_path}")
        return {}
    if genai is None:
        print("  ✗ google-genai package not installed.")
        return {}

    print(f"  Mapping terminology: {stem}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        segments = data.get("segments", [])
        detected_language = data.get("language", "unknown")
    elif isinstance(data, list):
        segments = data
        detected_language = "unknown"
    else:
        print(f"  ⚠ Unexpected format in {transcript_path}")
        return {}

    # Filter to code-mixed segments only
    cm_segments = []
    cm_indices = []
    for i, seg in enumerate(segments):
        if _is_code_mixed(seg):
            cm_segments.append({
                "index": i,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "original_text": seg.get("original_text", ""),
                "english_text": seg.get("english_text", ""),
                "language": seg.get("language", detected_language),
            })
            cm_indices.append(i)

    if not cm_segments:
        print(f"  ⚠ No code-mixed segments found — skipping mapping.")
        return _build_result(stem, detected_language, len(segments), [])

    # ── Batch segments for Gemini (max ~15 segments per call to stay
    #    within context limits) ────────────────────────────────────────
    BATCH_SIZE = 15
    all_mappings: list[dict] = []
    client = genai.Client(api_key=api_key)

    for batch_start in range(0, len(cm_segments), BATCH_SIZE):
        batch = cm_segments[batch_start : batch_start + BATCH_SIZE]

        # Build prompt with batch data
        batch_payload = []
        for seg in batch:
            batch_payload.append({
                "segment_index": seg["index"],
                "start": seg["start"],
                "end": seg["end"],
                "original_text": seg["original_text"],
                "english_text": seg["english_text"],
                "language": seg["language"],
            })

        prompt = _MAPPING_PROMPT + "\n\nSegments:\n" + json.dumps(batch_payload, indent=2, ensure_ascii=False)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )

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

            parsed = _extract_json(raw)
            if parsed is None:
                print(f"  ⚠ JSON parse failed for batch starting at segment {batch[0]['index']}")
                continue

            # Extract the mappings array
            batch_mappings = []
            if isinstance(parsed, dict):
                batch_mappings = parsed.get("mappings", [])
            elif isinstance(parsed, list):
                batch_mappings = parsed

            all_mappings.extend(batch_mappings)

        except Exception as e:
            print(f"  ⚠ Gemini call failed for batch: {e}")
            continue

        # Rate-limit between batches
        if batch_start + BATCH_SIZE < len(cm_segments):
            sleep(10)

    # ── Post-process: clean up and validate entries ──────────────────
    cleaned_mappings = _clean_mappings(all_mappings)

    result = _build_result(stem, detected_language, len(segments), cleaned_mappings)

    # ── Write outputs ────────────────────────────────────────────────
    # 1. data/terminology_mappings/<stem>.json
    mapping_out = MAPPING_DIR / f"{stem}.json"
    with open(mapping_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Terminology mapping → {mapping_out}")

    # 2. outputs/<stem>/terminology_mapping.json
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    output_copy = out_dir / "terminology_mapping.json"
    with open(output_copy, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Output copy         → {output_copy}")

    return result


def _build_result(stem: str, language: str, total_segments: int,
                  mappings: list[dict]) -> dict:
    """Assemble the final result dict."""
    total_entries = sum(len(m.get("entries", [])) for m in mappings)
    mapped_count = sum(1 for m in mappings if m.get("entries"))
    return {
        "video": stem,
        "language": language,
        "total_segments": total_segments,
        "mapped_segments": mapped_count,
        "total_mappings": total_entries,
        "mappings": mappings,
    }


def _clean_mappings(mappings: list[dict]) -> list[dict]:
    """Validate and deduplicate mapping entries."""
    seen = set()
    cleaned = []

    for seg_map in mappings:
        if not isinstance(seg_map, dict):
            continue
        entries = seg_map.get("entries", [])
        if not isinstance(entries, list):
            continue

        clean_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            orig = entry.get("original_phrase", "").strip()
            std = entry.get("standard_term", "").strip()
            if not orig or not std:
                continue
            # Deduplicate by (original_phrase, standard_term)
            key = (orig.lower(), std.lower())
            if key in seen:
                continue
            seen.add(key)

            # Normalise the category field
            cat = entry.get("category", "technical").strip().lower()
            if cat not in ("technical", "conversational"):
                cat = "technical"

            clean_entries.append({
                "original_phrase": orig,
                "language": entry.get("language", "unknown").strip(),
                "literal_translation": entry.get("literal_translation", "").strip(),
                "standard_term": std,
                "category": cat,
                "context": entry.get("context", "").strip(),
            })

        cleaned.append({
            "segment_index": seg_map.get("segment_index", -1),
            "start": float(seg_map.get("start", 0.0)),
            "end": float(seg_map.get("end", 0.0)),
            "entries": clean_entries,
        })

    return cleaned


# ── CLI entry point ──────────────────────────────────────────────────────

def process_all():
    """Process all transcript files in data/transcripts/."""
    if sys.argv[1:]:
        # Process a single file
        path = sys.argv[1]
        generate_mapping(path)
        return

    transcript_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))
    if not transcript_files:
        print("No transcript JSON files found in", TRANSCRIPT_DIR)
        return

    print(f"Terminology Mapping: {len(transcript_files)} files")
    for fpath in transcript_files:
        generate_mapping(fpath)
        sleep(5)  # rate-limit between videos

    print("Terminology mapping complete.")


if __name__ == "__main__":
    process_all()
