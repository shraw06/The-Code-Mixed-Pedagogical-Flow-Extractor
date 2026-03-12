#!/usr/bin/env python3
"""
Step 5 – Concept Extraction (code-mixed aware).

Extracts technical concepts from cleaned transcripts produced by
linguistic_standardizer.py.  Each cleaned segment now carries:
  - text           : normalised English (for NLP extraction)
  - english_text   : full English translation (for KeyBERT context)
  - original_text  : verbatim code-mixed speech (for provenance)
  - start / end    : segment timestamps
  - language       : code-mix label

The module uses spaCy (noun chunks + NER) and KeyBERT to extract candidate
concepts, then consolidates near-duplicates and writes:
  - data/concepts/<stem>.json   (concept → time references mapping)

Usage:
    python pipeline/concept_extract.py
"""

import spacy
import glob
import os
import json
from pathlib import Path
from collections import defaultdict

from keybert import KeyBERT

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned_transcripts"
CONCEPTS_DIR = ROOT / "data" / "concepts"
CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)

# Words that are too generic / off-topic for a technical concept
GENERIC_WORDS = {
    "video", "thing", "chapter", "example",
    "friend", "today", "watch", "look", "share",
    "lecture", "question", "answer", "sir", "madam",
    "guys", "point", "topic", "minute", "second",
    "way", "time", "part", "number", "case",
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normalize_phrase(phrase: str) -> str:
    """Lemmatise + remove stop-words from a short phrase."""
    doc = nlp(phrase)
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)


def extract_concepts(text: str) -> list[str]:
    """Extract candidate concept phrases from a text string."""
    doc = nlp(text)
    concepts = set()

    # Noun chunks
    for chunk in doc.noun_chunks:
        phrase = normalize_phrase(chunk.text.lower())
        if 1 <= len(phrase.split()) <= 4:
            concepts.add(phrase)

    # Named entities
    for ent in doc.ents:
        phrase = normalize_phrase(ent.text.lower())
        if 1 <= len(phrase.split()) <= 4:
            concepts.add(phrase)

    # KeyBERT keywords (broader context)
    try:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=20,
        )
        for kw, _score in keywords:
            phrase = normalize_phrase(kw.lower())
            if 1 <= len(phrase.split()) <= 4:
                concepts.add(phrase)
    except Exception:
        pass  # KeyBERT can fail on very short texts

    # Filter out generic words
    filtered = [
        c for c in concepts
        if not any(w in GENERIC_WORDS for w in c.split()) and c.strip()
    ]
    return sorted(filtered)


def normalize_concept(phrase: str) -> str:
    """Canonical normalisation of a concept phrase."""
    doc = nlp(phrase)
    return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)


def consolidate_concepts(concepts: list[str]) -> list[str]:
    """Merge near-duplicate concepts by grouping on their root noun."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in concepts:
        normed = normalize_concept(c)
        if not normed:
            continue
        root = normed.split()[-1]  # last word = main noun
        groups[root].append(normed)

    final = set()
    for _root, variants in groups.items():
        best = min(variants, key=len)
        final.add(best)
    return sorted(final)


# ─────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────

def process_all():
    # ── Process .json files (segment-level with timestamps) ──────────────
    for json_path in sorted(glob.glob(str(CLEANED_DIR / "*.json"))):
        stem = Path(json_path).stem
        print(f"Extracting concepts (json): {stem}")

        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        concept_occurrences: list[dict] = []

        for seg in segments:
            # Use the cleaned text for primary extraction
            cleaned_text = seg.get("text", "")
            # Also try extracting from the fuller english_text for additional concepts
            english_text = seg.get("english_text", cleaned_text)
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))

            # Extract from cleaned (lemmatised) text
            extracted = set(extract_concepts(cleaned_text))
            # Also extract from the richer english translation
            if english_text and english_text != cleaned_text:
                extracted.update(extract_concepts(english_text))

            for c in extracted:
                concept_occurrences.append({
                    "concept": c,
                    "start": start,
                    "end": end,
                })

        # Normalise and consolidate
        normalized_occurrences = []
        for occ in concept_occurrences:
            normed = normalize_concept(occ["concept"])
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

        # Write JSON output
        out = CONCEPTS_DIR / f"{stem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(final_concepts_data, f, indent=4, ensure_ascii=False)
        print(f"  ✓ {len(final_concepts_data)} concepts → {out}")


if __name__ == "__main__":
    process_all()