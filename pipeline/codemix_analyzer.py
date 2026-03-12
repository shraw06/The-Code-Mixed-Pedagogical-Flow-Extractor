#!/usr/bin/env python3
"""
Step 4b – Code-Mixing Analysis.

Quantitative and qualitative analysis of code-mixing patterns in each
video transcript.  This module operates on the Gemini-produced transcripts
(data/transcripts/*.json) which preserve the verbatim ``original_text``
alongside the ``english_text`` translation and the per-segment ``language``
label.

Metrics computed per video:
  • CMI  (Code-Mixing Index) — fraction of non-matrix-language words
  • Language distribution — word-count and segment-count per language
  • Switch-point density — how often the speaker switches languages
    within a segment (approximated via script detection)
  • Matrix vs. embedded language classification
  • Segment-level language-tag timeline (for downstream visualisation)

Outputs (per video):
  data/analysis/<stem>_codemix.json   — full metrics + per-segment breakdown

Usage:
    python pipeline/codemix_analyzer.py
"""

import json
import glob
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
CLEANED_DIR = ROOT / "data" / "cleaned_transcripts"
ANALYSIS_DIR = ROOT / "data" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Script / language detection helpers
# ─────────────────────────────────────────────

def _char_script(ch: str) -> str:
    """Classify a single character by Unicode script block."""
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return "OTHER"
    name = name.upper()
    if "DEVANAGARI" in name:
        return "DEVANAGARI"      # Hindi, Marathi
    elif "TAMIL" in name:
        return "TAMIL"
    elif "TELUGU" in name:
        return "TELUGU"
    elif "KANNADA" in name:
        return "KANNADA"
    elif "MALAYALAM" in name:
        return "MALAYALAM"
    elif "BENGALI" in name or "BANGLA" in name:
        return "BENGALI"
    elif "GURMUKHI" in name:
        return "GURMUKHI"        # Punjabi
    elif "GUJARATI" in name:
        return "GUJARATI"
    elif "LATIN" in name or ch.isascii() and ch.isalpha():
        return "LATIN"
    else:
        return "OTHER"


def word_script(word: str) -> str:
    """Determine the dominant script of a word (by character majority)."""
    counts: Counter = Counter()
    for ch in word:
        if ch.isalpha():
            counts[_char_script(ch)] += 1
    if not counts:
        return "OTHER"
    return counts.most_common(1)[0][0]


def is_indic_script(script: str) -> bool:
    return script in {
        "DEVANAGARI", "TAMIL", "TELUGU", "KANNADA",
        "MALAYALAM", "BENGALI", "GURMUKHI", "GUJARATI",
    }


# ─────────────────────────────────────────────
# Token-level analysis of original_text
# ─────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation-aware tokeniser."""
    return [w for w in re.findall(r"\w+", text, re.UNICODE) if len(w) > 0]


def analyze_segment_mixing(original_text: str):
    """Analyse code-mixing within a single segment.

    Returns a dict with:
      words         : list of (word, script)
      script_counts : Counter of scripts
      n_switches    : number of language switch-points
      cmi           : segment-level CMI (0..1)
      matrix_script : most-frequent script in this segment
    """
    words = tokenize(original_text)
    if not words:
        return {
            "words": [], "script_counts": Counter(),
            "n_switches": 0, "cmi": 0.0, "matrix_script": "UNKNOWN",
        }

    tagged = [(w, word_script(w)) for w in words]
    scripts = Counter(s for _, s in tagged if s != "OTHER")

    # Matrix language = most frequent script (by word count)
    if scripts:
        matrix = scripts.most_common(1)[0][0]
    else:
        matrix = "LATIN"

    # Switch points: consecutive words in different scripts (ignoring OTHER)
    prev_script = None
    n_switches = 0
    for _, s in tagged:
        if s == "OTHER":
            continue
        if prev_script is not None and s != prev_script:
            n_switches += 1
        prev_script = s

    # CMI = (N - max_script_count) / N  (standard definition)
    total = sum(scripts.values())
    if total > 1 and scripts:
        max_count = scripts.most_common(1)[0][1]
        cmi = (total - max_count) / total
    else:
        cmi = 0.0

    return {
        "words": tagged,
        "script_counts": dict(scripts),
        "n_switches": n_switches,
        "cmi": round(cmi, 4),
        "matrix_script": matrix,
    }


# ─────────────────────────────────────────────
# Romanised code-mixing detection
# ─────────────────────────────────────────────
# Many Indian educational videos use *romanised* Indic languages (Hindi
# written in Latin script).  Pure script-detection misses these.  We use
# the language label from Gemini + a simple heuristic: if all words are
# LATIN but the label says "Hindi-English", we detect romanised mixing.

def detect_romanised_mixing(original_text: str, language_label: str):
    """Heuristic for romanised code-mixing.

    If all tokens are Latin script but the language label indicates mixing
    (e.g. "Hindi-English"), we infer romanised code-mixing is present.

    Returns:
      is_romanised_mix : bool
      estimated_indic_fraction : float (0..1) — rough estimate via the
          language label; 0.5 if label says "<L>-English" else 0.
    """
    words = tokenize(original_text)
    if not words:
        return False, 0.0

    scripts = [word_script(w) for w in words]
    all_latin = all(s in ("LATIN", "OTHER") for s in scripts)

    lang_lower = language_label.lower()
    mix_indicators = [
        "hindi", "tamil", "telugu", "marathi", "kannada",
        "malayalam", "bengali", "gujarati", "punjabi",
    ]
    has_indic_label = any(ind in lang_lower for ind in mix_indicators)

    if all_latin and has_indic_label:
        # Rough estimate: if label is "X-English", assume ~50% Indic
        return True, 0.5
    return False, 0.0


# ─────────────────────────────────────────────
# Video-level analysis
# ─────────────────────────────────────────────

def analyze_video(transcript_path: str) -> dict:
    """Full code-mixing analysis for a single video transcript."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        segments = data.get("segments", [])
    elif isinstance(data, list):
        segments = data
    else:
        return {"error": "unexpected format"}

    seg_analyses = []
    total_words = 0
    total_switches = 0
    global_script_counts: Counter = Counter()
    language_labels: Counter = Counter()
    romanised_segments = 0

    for seg in segments:
        original = seg.get("original_text", seg.get("text", ""))
        lang = seg.get("language", "unknown")
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))

        # Script-based analysis
        sa = analyze_segment_mixing(original)

        # Romanised detection
        is_rom, rom_frac = detect_romanised_mixing(original, lang)
        if is_rom:
            romanised_segments += 1

        n_words = len(sa["words"])
        total_words += n_words
        total_switches += sa["n_switches"]
        global_script_counts.update(sa["script_counts"])
        language_labels[lang] += 1

        seg_analyses.append({
            "start": start,
            "end": end,
            "language": lang,
            "n_words": n_words,
            "script_counts": sa["script_counts"],
            "n_switches": sa["n_switches"],
            "cmi": sa["cmi"],
            "matrix_script": sa["matrix_script"],
            "is_romanised_mix": is_rom,
        })

    # Global CMI
    if global_script_counts:
        total_scripted = sum(global_script_counts.values())
        max_script = global_script_counts.most_common(1)[0][1]
        global_cmi = (total_scripted - max_script) / total_scripted if total_scripted > 1 else 0.0
    else:
        global_cmi = 0.0

    # Matrix language
    matrix_script = global_script_counts.most_common(1)[0][0] if global_script_counts else "LATIN"

    # Switch-point density (switches per 100 words)
    switch_density = (total_switches / total_words * 100) if total_words > 0 else 0.0

    # Identify embedded language(s) — all scripts that are NOT the matrix
    embedded_scripts = {s: c for s, c in global_script_counts.items() if s != matrix_script}

    # Determine if mostly romanised
    is_mostly_romanised = romanised_segments > len(segments) * 0.5

    # Map script back to likely language name
    script_to_lang = {
        "DEVANAGARI": "Hindi/Marathi",
        "TAMIL": "Tamil",
        "TELUGU": "Telugu",
        "KANNADA": "Kannada",
        "MALAYALAM": "Malayalam",
        "BENGALI": "Bengali",
        "GURMUKHI": "Punjabi",
        "GUJARATI": "Gujarati",
        "LATIN": "English/Romanised",
    }

    return {
        "video": Path(transcript_path).stem,
        "total_segments": len(segments),
        "total_words": total_words,
        "global_cmi": round(global_cmi, 4),
        "switch_density_per_100_words": round(switch_density, 2),
        "matrix_script": matrix_script,
        "matrix_language": script_to_lang.get(matrix_script, matrix_script),
        "embedded_scripts": {s: c for s, c in sorted(embedded_scripts.items(), key=lambda x: -x[1])},
        "is_mostly_romanised": is_mostly_romanised,
        "romanised_segment_fraction": round(romanised_segments / max(len(segments), 1), 4),
        "language_label_distribution": dict(language_labels.most_common()),
        "script_distribution": dict(global_script_counts.most_common()),
        "segment_analyses": seg_analyses,
    }


def _human_summary(result: dict) -> str:
    """Produce a human-readable summary string."""
    lines = []
    lines.append(f"Video: {result['video']}")
    lines.append(f"  Segments: {result['total_segments']}")
    lines.append(f"  Total words (tokenised): {result['total_words']}")
    lines.append(f"  Global CMI (Code-Mixing Index): {result['global_cmi']}")
    lines.append(f"  Switch-point density: {result['switch_density_per_100_words']} per 100 words")
    lines.append(f"  Matrix script: {result['matrix_script']} ({result['matrix_language']})")
    lines.append(f"  Embedded scripts: {result['embedded_scripts']}")
    lines.append(f"  Mostly romanised code-mixing: {'Yes' if result['is_mostly_romanised'] else 'No'}")
    lines.append(f"    Romanised segment fraction: {result['romanised_segment_fraction']}")
    lines.append(f"  Language labels (from Gemini): {result['language_label_distribution']}")
    lines.append(f"  Script distribution: {result['script_distribution']}")

    # Per-segment CMI histogram (text-based)
    cmis = [s["cmi"] for s in result["segment_analyses"]]
    if cmis:
        avg_cmi = sum(cmis) / len(cmis)
        lines.append(f"  Avg segment CMI: {avg_cmi:.4f}")
        high_mix = sum(1 for c in cmis if c > 0.3)
        lines.append(f"  Highly code-mixed segments (CMI > 0.3): {high_mix}/{len(cmis)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def process_all():
    transcript_files = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.json")))
    if not transcript_files:
        print("No transcript JSON files found in", TRANSCRIPT_DIR)
        return

    all_results = []

    for fpath in transcript_files:
        stem = Path(fpath).stem
        print(f"Analysing code-mixing: {stem}")

        result = analyze_video(fpath)
        all_results.append(result)

        # Save per-video JSON
        json_out = ANALYSIS_DIR / f"{stem}_codemix.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ JSON → {json_out}")

    # ── Cross-video comparison ──
    if len(all_results) > 1:
        comparison = {
            "videos": [],
            "ranking_by_cmi": [],
        }
        for r in all_results:
            comparison["videos"].append({
                "video": r["video"],
                "global_cmi": r["global_cmi"],
                "switch_density": r["switch_density_per_100_words"],
                "matrix_language": r["matrix_language"],
                "is_mostly_romanised": r["is_mostly_romanised"],
                "total_words": r["total_words"],
                "language_labels": r["language_label_distribution"],
            })
        comparison["ranking_by_cmi"] = sorted(
            [{"video": r["video"][:60], "cmi": r["global_cmi"]} for r in all_results],
            key=lambda x: -x["cmi"],
        )

        comp_path = ANALYSIS_DIR / "cross_video_codemix_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Cross-video comparison → {comp_path}")


if __name__ == "__main__":
    process_all()
