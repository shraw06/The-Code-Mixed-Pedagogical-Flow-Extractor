#!/usr/bin/env python3
"""
Step 4 – Linguistic Standardisation (code-mixed aware).

This module operates on the Gemini-produced transcripts that already contain
both ``original_text`` (verbatim code-mixed) and ``english_text`` (English
translation).  Its job:

  1. Clean ASR / LLM artifacts (extra whitespace, non-ASCII noise, repeated
     words).
  2. Light NLP normalisation (lemmatisation, stop-word removal) on the
     **english_text** to produce ``cleaned_text`` — used by concept extraction.
  3. Preserve the ``original_text`` and ``language`` metadata so that
     downstream analysis can reference the raw code-mixed phrasing.
  4. Filter out segments that are too short or contain no informative nouns.
  5. Write cleaned transcript JSON + plain-text into data/cleaned_transcripts/.

The output JSON per segment:
  {
    "start": float,
    "end": float,
    "original_text": str,    # verbatim code-mixed speech
    "english_text": str,     # English translation (from STT step)
    "text": str,             # cleaned / normalised English (for concept extraction)
    "language": str          # code-mix label, e.g. "Hindi-English"
  }

Usage:
    python pipeline/linguistic_standardizer.py
"""

import json
import re
import glob
import os
from pathlib import Path

import spacy

nlp = spacy.load("en_core_web_sm")

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
CLEANED_DIR = ROOT / "data" / "cleaned_transcripts"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Clean ASR / LLM artefacts
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lower-case, strip non-ASCII noise, collapse whitespace."""
    text = text.lower()
    # remove non-ASCII characters (keeps romanised text fine)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────
# 2. Remove heavy word-level repetition
# ─────────────────────────────────────────────

def remove_repetition(text: str) -> str:
    words = text.split()
    filtered = []
    prev = None
    for w in words:
        if w != prev:
            filtered.append(w)
        prev = w
    return " ".join(filtered)


# ─────────────────────────────────────────────
# 3. Lemmatisation + stop-word removal
# ─────────────────────────────────────────────

def normalize_sentence(sentence: str) -> str:
    doc = nlp(sentence)
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)


# ─────────────────────────────────────────────
# Full standardisation pipeline (per segment)
# ─────────────────────────────────────────────

def linguistic_standardize_segment(text: str) -> str:
    """Apply the full cleaning pipeline to a single text string."""
    text = clean_text(text)
    text = remove_repetition(text)
    text = normalize_sentence(text)
    return text


# ─────────────────────────────────────────────
# Process all transcript files
# ─────────────────────────────────────────────

def process_all():
    transcript_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))
    if not transcript_files:
        print("No transcript JSON files found in", TRANSCRIPT_DIR)
        return

    for fpath in transcript_files:
        stem = Path(fpath).stem
        print(f"Standardising: {stem}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both old Whisper format and new Gemini format
        if isinstance(data, dict):
            segments = data.get("segments", [])
            detected_language = data.get("language", "unknown")
        elif isinstance(data, list):
            segments = data
            detected_language = "unknown"
        else:
            print(f"  ⚠ Unexpected format in {fpath}, skipping.")
            continue

        cleaned_segments = []

        for seg in segments:
            # The english_text field is the primary input for cleaning.
            # Fall back to "text" for backward compatibility with Whisper output.
            english_text = seg.get("english_text", seg.get("text", ""))
            original_text = seg.get("original_text", english_text)
            lang = seg.get("language", detected_language)

            cleaned = linguistic_standardize_segment(english_text)

            # Keep only segments that contain at least one noun (informative)
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

        # ── Save cleaned JSON ──
        json_out = CLEANED_DIR / f"{stem}.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(cleaned_segments, f, indent=4, ensure_ascii=False)
        print(f"  ✓ Cleaned JSON ({len(cleaned_segments)} segments) → {json_out}")


if __name__ == "__main__":
    process_all()