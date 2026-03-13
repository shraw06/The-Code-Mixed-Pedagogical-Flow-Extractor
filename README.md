# Code-Mixed Pedagogical Flow Extractor

> **iREL Recruitment Task 2026**  
> A robust NLP pipeline that processes raw, noisy code-mixed educational video content and extracts a formal, structured representation of pedagogical flow and concept dependencies.

---

## Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Architectural Overview](#architectural-overview)  
3. [Key Design Decisions](#key-design-decisions)  
4. [Pipeline Stages](#pipeline-stages)  
5. [Video Sources & Languages](#video-sources--languages)  
6. [Output Structure & Rationale](#output-structure--rationale)  
7. [Code-Mixing Analysis](#code-mixing-analysis)  
8. [Terminology Mapping](#terminology-mapping)
9. [Setup & Reproduction](#setup--reproduction)  
10. [Project Structure](#project-structure)

---

## Problem Statement

Indian educational content on platforms like YouTube is incredibly rich but highly unstructured. Teachers freely **code-mix** - switching between English and regional languages (Hindi, Tamil, Telugu, Marathi, etc.) - while teaching technical concepts. This makes computational analysis extremely challenging.

This pipeline ingests 5 such code-mixed educational videos, extracts the core technical concepts, maps colloquial terms to standard English academic terminology, and builds a **directed acyclic graph (DAG)** representing the pedagogical prerequisite flow - which concepts must be understood before others. It additionally computes a **topologically sorted learning roadmap** and provides quantitative **code-mixing analysis** for each video.

---

## Architectural Overview

The pipeline is designed as a **linear chain of self-contained stages**, each consuming the outputs of the previous stage and writing to well-defined directories. This design allows any stage to be re-run independently and makes debugging straightforward - intermediate results in `data/` are always inspectable.

```
YouTube Videos / Uploaded File
           │
           ▼
  ┌──────────────┐
  │ 1. Download  │  yt-dlp (skipped if file uploaded)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 2. Extract   │  ffmpeg → 16 kHz mono WAV
  │    Audio     │
  └──────┬───────┘
         ▼
  ┌──────────────────────────────────────────────────────┐
  │ 3. Speech-to-Text (Gemini 2.5 Flash)                 │
  │    • Transcription-first                             │
  │    • Preserves verbatim code-mixed original_text     │
  │    • Parallel English translation per segment        │
  │    • Per-segment language label                      │
  │    • Timestamped segments                            │
  └──────┬───────────────────────────────────────────────┘
         ▼
  ┌──────────────────────────────────────────────────────┐
  │ 4. Linguistic Standardisation                        │
  │    • Clean ASR artefacts (noise, repetition)         │
  │    • spaCy lemmatisation + stop-word removal         │
  │    • Preserve original_text + language metadata      │
  │    • Filter non-informative segments (noun check)    │
  └──────┬───────────────────────────────────────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  ┌──────────────────┐        ┌──────────────────────────┐
  │ 5. Concept       │        │ 4b. Code-Mixing Analysis │
  │    Extraction    │        │     • CMI computation     │
  │    • spaCy NER   │        │     • Script detection    │
  │    • Noun chunks │        │     • Switch-point density│
  │    • KeyBERT     │        │     • Romanised detection │
  └──────┬───────────┘        └──────────┬───────────────┘
         ▼                               │
  ┌──────────────────────────────────┐   │
  │ 6. Prerequisite Graph (Gemini)   │   │
  │    • LLM-powered DAG             │   │
  │    • Pedagogical ordering        │   │
  │    • Concept refinement/merging  │   │
  └──────┬───────────────────────────┘   │
         │                               │
         ├───────────────────────────┐   │
         ▼                           │   │
  ┌──────────────────┐               │   │
  │ 7. Visualisation │               │   │
  │  • Interactive   │               │   │
  │    HTML (pyvis)  │               │   │
  │  • Static PNG    │               │   │
  │  • GEXF export   │               │   │
  └──────┬───────────┘               │   │
         ▼                           │   │
  ┌──────────────────┐               │   │
  │ 8. Learning      │               │   │
  │    Roadmap       │               │   │
  │  • Topological   │               │   │
  │    sort + levels │               │   │
  │  • Flowchart PNG │               │   │
  └──────────────────┘               │   │
                                     │   │
                       ┌─────────────┘   │
                       ▼                 ▼
              ┌────────────────────┐
              │ 9. Code-Mix       │
              │    Visualisation  │
              │  • CMI profile    │
              │  • Script dist.   │
              └────────────────────┘
```

### Why This Architecture?

| Decision | Rationale |
|----------|-----------|
| **Linear stage chain** | Each stage reads from disk and writes to disk. This means any stage can be re-run without replaying the full pipeline - essential during development and debugging. |
| **Subprocess-per-stage in `main.py`** | Avoids side-effects from modules that execute work at import time. Each stage runs in its own Python process for isolation. |
| **Gemini for transcription AND graph construction** | A single LLM provider simplifies API-key management and gives consistent behaviour. Gemini 2.5 Flash excels at both audio understanding and structured-JSON generation. |
| **JSON-only outputs** | Every intermediate and final artefact is JSON (except visualisation images). JSON is human-readable, machine-parseable, and diff-friendly — ideal for reproducibility. |
| **Per-video output directories** | Keeps outputs self-contained: each directory under `outputs/` holds everything needed to understand one video's pedagogical structure. |

---

## Key Design Decisions

### 1. Transcription-First, Not Translation-First

**The most critical architectural decision.** Initially the approach used OpenAI Whisper in `translate` mode, which directly translates code-mixed speech into English - **destroying the original code-mixed phrasing entirely**. This loses:
- The ability to analyse *which language* each concept was taught in
- Colloquial Indic terms and analogies used to explain technical concepts
- The matrix/embedded language structure

Our pipeline uses Gemini's multimodal capabilities to **transcribe first** - preserving the verbatim code-mixed speech - then provides a parallel English translation per segment. Every segment carries:

```json
{
  "start": 0.0,
  "end": 12.5,
  "original_text": "Hello dosto, aaj ka video wakiye bada kamaal ka hone wala hai...",
  "english_text": "Hello friends, today's video is going to be really amazing...",
  "language": "Hindi-English"
}
```

This dual representation enables downstream code-mixing analysis on `original_text` while using `english_text` for concept extraction and graph building.

### 2. Gemini API Over Whisper

- **Whisper** struggles with heavily code-mixed audio and has no native Indic-script transcription
- **Gemini 2.5 Flash** handles code-mixed audio natively, transcribes in the original script (Devanagari, Telugu, Tamil) or romanised form, and provides structured JSON output
- `temperature=0` + `response_mime_type="application/json"` ensures deterministic, parseable output with no hallucinated structure
- Audio files uploaded via the Files API to handle large lectures (10-30 minutes)

### 3. Dual-Track Concept Extraction

Concepts are extracted from both the cleaned/lemmatised English text (for precision) and the richer `english_text` translation (for recall), then consolidated via deduplication. This captures:
- Concepts that survive aggressive lemmatisation/stop-word removal
- Concepts that appear only in the fuller English translation

The combination of **spaCy** (noun chunks + NER) and **KeyBERT** (semantic keyphrases using `all-MiniLM-L6-v2`) yields higher coverage than either alone.

### 4. LLM-Powered Prerequisite Graph

Rather than using heuristic co-occurrence or temporal proximity alone, we use Gemini to understand **pedagogical intent** - which concepts the teacher presents as prerequisites for others. The LLM receives:
- Concept names + timestamp ranges
- The detected code-mixed language
- Instructions to build a DAG with transitive reduction

This produces a clean prerequisite graph that reflects the teacher's intended learning order, not just statistical co-occurrence.

### 5. Code-Mixing as a First-Class Signal

Code-mixing is not just noise to be cleaned away - it's a **signal** about how educators teach. We compute:
- **CMI (Code-Mixing Index)** — standard sociolinguistic metric
- **Script distribution** — Devanagari vs Latin vs Telugu etc.
- **Romanised detection** — many videos use Latin-script Indic languages
- **Switch-point density** — how frequently the speaker alternates languages

### 6. Topological Learning Roadmap

The prerequisite graph is topologically sorted (with automatic cycle-breaking via frequency heuristics) and concepts are assigned to **levels** — all concepts at the same level can be studied in parallel once their prerequisites are complete. This produces a concrete study plan, not just a dependency graph.

---

## Pipeline Stages

| # | Stage | Script | Input → Output |
|---|-------|--------|-----------------|
| 1 | Video Download | `video_downloader.py` | YouTube URLs -> MP4 files (`data/videos/`) |
| 2 | Audio Extraction | `audio_extractor.py` | MP4 -> 16 kHz mono WAV (`data/audios/`) |
| 3 | Speech-to-Text | `speech_to_text.py` | WAV -> timestamped code-mixed + English JSON (`data/transcripts/`) |
| 4 | Linguistic Standardisation | `linguistic_standardizer.py` | Raw transcripts -> cleaned, lemmatised segments (`data/cleaned_transcripts/`) |
| 4b | Code-Mix Analysis | `codemix_analyzer.py` | Cleaned transcripts -> CMI, script stats, switch density (`data/analysis/`) |
| 5 | Concept Extraction | `concept_extract.py` | Cleaned transcripts -> concepts with timestamps (`data/concepts/`) |
| 6 | Prerequisite Graph | `build_graph.py` | Concepts -> LLM-powered pedagogical DAG (`data/graphs/`) |
| 7 | Visualisation | `visualize_graph.py` | Graph JSON -> interactive HTML, static PNG, GEXF, canonical JSON (`outputs/`) |
| 8 | Learning Roadmap | `topological_sort.py` | Canonical JSON -> topologically sorted study plan + flowchart PNG (`outputs/`) |
| 9 | Code-Mix Visualisation | `visualize_codemix.py` | Analysis JSON -> CMI profile + language distribution chart (`outputs/`) |

---

## Video Sources & Languages

The pipeline was tested on **5 educational videos** spanning **4 Indic languages**, **3 scripts**, and **5 technical domains** to demonstrate robustness across diverse code-mixing styles.

| # | Video Title | Language Pair | Script | Domain | YouTube Link |
|---|------------|---------------|--------|--------|--------------|
| 1 | Eigen Values & Eigen Vectors \| Matrices \| Numericals \| Maths | **Hindi–English** | Devanagari + Latin | Linear Algebra | [youtube.com/watch?v=BdRPgPfLAUQ](https://www.youtube.com/watch?v=BdRPgPfLAUQ) |
| 2 | What is Ohm's Law \| Explained in Telugu | **Telugu–English** | Telugu + Latin | Physics (Circuits) | [youtube.com/watch?v=XrfsQrTJ8Vo](https://www.youtube.com/watch?v=XrfsQrTJ8Vo) |
| 3 | What is Linked List \| Data Structure (Hindi) | **Hindi–English** | Romanised Latin | Data Structures | [youtube.com/watch?v=ESQfsAKQrJ4](https://www.youtube.com/watch?v=ESQfsAKQrJ4) |
| 4 | What is Python \| History of Python (Tamil) | **Tamil–English** | Romanised Latin | Programming | [youtube.com/watch?v=dgZECmAqE7w](https://www.youtube.com/watch?v=dgZECmAqE7w) |
| 5 | Introduction to DBMS in Marathi | **Marathi–English** | Devanagari + Latin | Databases | [youtube.com/watch?v=PurIKzKlUso](https://www.youtube.com/watch?v=PurIKzKlUso) |

### Language & Script Diversity

| Dimension | Coverage |
|-----------|----------|
| **Indic languages** | Hindi, Tamil, Telugu, Marathi |
| **Scripts** | Devanagari (Hindi, Marathi), Telugu script, Latin (romanised Hindi, Tamil, English) |
| **Code-mixing styles** | Native-script mixing (videos 1, 2, 5) and romanised mixing (videos 3, 4) |
| **Technical domains** | Linear Algebra, Physics, Data Structures, Programming, Databases |

This selection was deliberate: it covers both **native-script code-mixing** (where the teacher writes/speaks in Devanagari or Telugu alongside English) and **romanised code-mixing** (where the teacher uses Latin script for both the Indic language and English). These two styles require fundamentally different detection strategies — script-based CMI works well for the former but fails for the latter, motivating our multi-layered analysis approach.

---

## Output Structure & Rationale

### Design Rationale

Every video produces a self-contained output directory under `outputs/<video>/`. The output formats were chosen to serve distinct audiences:

| Audience | Files | Why |
|----------|-------|-----|
| **Downstream ML pipelines** | `canonical.json`, `learning_roadmap.json` | Strictly validated, machine-readable JSON — easy to ingest into graph databases, recommendation systems, or further NLP analysis |
| **Human reviewers / domain experts** | `interactive.html`, `dependency.png`, `learning_roadmap.png` | Visual representations that let a human quickly verify the extracted pedagogy without writing code |
| **Graph analysis tools** (Gephi, igraph) | `<video>.gexf` | Standard GEXF format for import into dedicated graph-analysis software |
| **Sociolinguistic analysis** | `codemix_profile.png`, `data/analysis/*.json` | Quantitative code-mixing metrics + visual summaries for language researchers |

No `.txt` files are generated — all structured data is JSON, and all visual data is HTML or PNG. This eliminates redundant formats and keeps the output directory clean.

### Per-Video Outputs (`outputs/<video>/`)

| File | Format | Description |
|------|--------|-------------|
| `canonical.json` | JSON | Strictly validated machine-readable graph (nodes + edges with timestamps, confidence, order) |
| `<video>.gexf` | GEXF | Standard graph exchange format (importable into Gephi, igraph) |
| `interactive.html` | HTML | Force-directed interactive graph with hover tooltips (pyvis) |
| `dependency.png` | PNG | Hierarchical static dependency diagram with colour-coded timeline |
| `learning_roadmap.json` | JSON | Topologically sorted concept order grouped by prerequisite levels |
| `learning_roadmap.png` | PNG | Layered flowchart visualising the study plan with colour-coded levels and prerequisite arrows |
| `codemix_profile.png` | PNG | Script distribution pie chart + CMI over time |

### Canonical JSON Schema

```json
{
  "video": "Video Title",
  "nodes": [
    {
      "id": "linked list",
      "concept": "linked list",
      "start": 5.2,
      "end": 142.7,
      "avg_t": 68.3,
      "freq": 12,
      "time_references": [{"start": 5.2, "end": 15.0}, ...]
    }
  ],
  "edges": [
    {
      "source": "data structure",
      "target": "linked list",
      "weight": 1,
      "explanation": "A linked list is a type of data structure."
    }
  ]
}
```

### Learning Roadmap JSON Schema

The `learning_roadmap.json` provides a topologically sorted ordering of concepts — a **study plan** where every prerequisite appears before the concepts that depend on it. Concepts are grouped into numbered **levels**: all concepts at the same level can be studied in parallel.

```json
{
  "video": "Video Title",
  "total_concepts": 13,
  "levels": [
    {
      "level": 1,
      "concepts": [
        {
          "concept": "Data Structure",
          "prerequisites": [],
          "first_mentioned": 0.0,
          "frequency": 5,
          "explanation": "Foundational concept — no prerequisites."
        }
      ]
    },
    {
      "level": 2,
      "concepts": [
        {
          "concept": "Linked List",
          "prerequisites": ["Data Structure"],
          "first_mentioned": 10.2,
          "frequency": 23,
          "explanation": "Linked List is a fundamental type of Data Structure."
        }
      ]
    }
  ],
  "linear_order": ["Data Structure", "Linked List", "..."]
}
```

### Intermediate Data (`data/`)

| Directory | Contents |
|-----------|----------|
| `data/transcripts/` | Raw Gemini transcripts (`.json` + `.raw_debug.txt`) |
| `data/cleaned_transcripts/` | Standardised segments (`.json`) |
| `data/concepts/` | Extracted concepts with timestamps (`.json`) |
| `data/graphs/` | LLM-generated prerequisite graphs (`.json` + `.raw.txt`) |
| `data/analysis/` | Code-mixing analysis (`.json` per video) |

---

## Code-Mixing Analysis

### Methodology

We use a multi-layered approach to quantify code-mixing:

1. **Unicode Script Detection**: Each word is classified by its dominant Unicode script block (Devanagari, Tamil, Telugu, Latin, etc.)

2. **Code-Mixing Index (CMI)**: Standard metric from sociolinguistics:
   
   $$CMI = \frac{N - \max(w_i)}{N}$$
   
   where $N$ is the total number of words and $\max(w_i)$ is the count of the most frequent language's words.

3. **Romanised Code-Mixing Detection**: Many Indian educators use Latin script for Indic languages (e.g., "Aaj ka video" instead of "आज का वीडियो"). We detect this by cross-referencing the Gemini language label with the observed script distribution.

4. **Switch-Point Density**: Number of language switches per 100 words, indicating how frequently the speaker alternates between languages.

### Key Findings

| Video | Matrix Language | CMI | Romanised? | Switch Density |
|-------|----------------|-----|------------|----------------|
| Eigen Values (Hindi) | Devanagari | 0.043 | No | 6.77/100w |
| Linked List (Hindi) | Latin | 0.000 | **Yes** (97%) | 0.08/100w |
| Python (Tamil) | Latin | 0.000 | **Yes** (100%) | 0.00/100w |
| Ohm's Law (Telugu) | Telugu | 0.001 | No | 0.06/100w |
| DBMS (Marathi) | Devanagari | 0.005 | No | 0.02/100w |

**Insight**: The script-based CMI reveals a dichotomy — videos transcribed in native scripts (Devanagari, Telugu) show measurable intra-sentence code-mixing, while romanised videos (Hindi in Latin, Tamil in Latin) have near-zero script-CMI because the Indic content is romanised. This highlights the importance of not relying solely on script detection for code-mixing analysis of Indian educational content.

---

## Terminology Mapping

This project also includes an explicit terminology-mapping stage which aligns colloquial Indic phrases and code-mixed analogies to standardized English academic terminology. Terminology mapping is important because teachers often use regional words, metaphors or shorthand that do not directly match textbook terms - mapping preserves traceability from the original lecture wording to canonical concepts used in the graphs and learning roadmap.

### What it does

- Identifies phrase-level correspondences between `original_text` (verbatim code-mixed transcription) and `english_text` (faithful English translation).  
- Produces a per-segment list of mappings: original phrase, literal translation, standardized English term, category ("technical" or "conversational"), and short context.  
- Outputs are written to `data/terminology_mappings/<video_stem>.json` and copied to `outputs/<video_stem>/terminology_mapping.json` so they are bundled with other per-video artifacts.

### Files & format

- Script: `pipeline/terminology_mapper.py` — runs the mapping stage (batching segments and calling Gemini).  
- Input: `data/transcripts/<video_stem>.json` (timestamped segments with `original_text` and `english_text`).  
- Primary outputs:
  - `data/terminology_mappings/<video_stem>.json` — authoritative mapping used by downstream stages.  
  - `outputs/<video_stem>/terminology_mapping.json` — a copy included in the per-video output bundle.

Example mapping (extract):

```json
{
  "mappings": [
    {
      "segment_index": 3,
      "start": 12.5,
      "end": 22.0,
      "entries": [
        {
          "original_phrase": "diye gaye matrix",
          "language": "Hindi",
          "literal_translation": "given matrix",
          "standard_term": "Given Matrix",
          "category": "technical",
          "context": "...in the original segment surrounding text..."
        }
      ]
    }
  ]
}
```

### How it works

- `terminology_mapper.py` filters to segments that appear code-mixed and batches them (≈15 segments per LLM call).  
- It prompts Gemini to produce a JSON array of mappings (the module robustly extracts JSON from the LLM response and performs validation + deduplication).  
- The module marks mappings as `technical` vs `conversational` and normalises entries to a compact, machine-readable schema.

### How to run

Run the mapping stage independently (recommended when adding or reviewing mappings):

```bash
python pipeline/terminology_mapper.py            # process all transcripts (requires GEMINI_API_KEY)
python pipeline/terminology_mapper.py data/transcripts/<video_stem>.json  # process one transcript
```

Notes:
- The stage requires a valid `GEMINI_API_KEY` in environment or `.env`.  
- If a valid mapping already exists for a video (non-empty mappings list) the script will skip regeneration.


### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| ffmpeg | any recent | Audio extraction from video files |
| Gemini API key | — | Transcription (Stage 3) and graph construction (Stage 6) |

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd irel_task

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install Python dependencies
pip install -r requirements.txt

# Download the spaCy language model
python -m spacy download en_core_web_sm

# Set your Gemini API key
echo "GEMINI_API_KEY=your-key-here" > .env
```

> **Note**: The pipeline uses `google-genai` (the new unified Google AI SDK), not the older `google-generativeai` package.

### Running the Full Pipeline (CLI)

```bash
python main.py
```

This runs all 10 stages sequentially for every video listed in `data/video_links.txt`. Individual stages can also be run independently:

```bash
python pipeline/speech_to_text.py       # Just transcription
python pipeline/codemix_analyzer.py     # Just code-mixing analysis
python pipeline/visualize_graph.py      # Just graph visualisation
python pipeline/topological_sort.py     # Just learning roadmap generation
```

### Running the Web UI

A Flask-based web interface is also available for single-video processing with real-time progress updates:

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser. The web UI supports:
- **File upload**: Upload a video file directly
- **YouTube URL**: Paste a YouTube link and the pipeline will download it automatically
- **Per-stage progress**: Server-Sent Events (SSE) stream logs and download links as each stage completes

### Dependencies

All Python dependencies are listed in `requirements.txt`:

```
yt-dlp              # Video downloading from YouTube
google-genai         # Gemini API client (transcription + graph building)
python-dotenv        # .env file loading for API key
spacy               # NLP: noun chunks, NER, lemmatisation
keybert              # Keyword extraction (all-MiniLM-L6-v2 embeddings)
networkx             # Graph data structures and algorithms
matplotlib           # Static visualisations (PNG charts, roadmap flowchart)
pyvis                # Interactive HTML graph generation
numpy                # Numerical operations for analysis metrics
flask                # Web UI server
```

---

## Project Structure

```
The-Code-Mixed-Pedagogical-Flow-Extractor/      # project root (repo)
├── main.py                          # CLI pipeline orchestrator (runs all stages)
├── app.py                           # Flask web UI (single-video processing with SSE)
├── requirements.txt                 # Python dependencies
├── .env                             # GEMINI_API_KEY (not committed)
├── README.md                        # This file
│
├── templates/
│   └── index.html                   # Web UI frontend (file upload + YouTube URL tabs)
│
├── pipeline/
│   ├── video_downloader.py          # Stage 1: yt-dlp video download
│   ├── audio_extractor.py           # Stage 2: ffmpeg audio extraction
│   ├── speech_to_text.py            # Stage 3: Gemini transcription
│   ├── linguistic_standardizer.py   # Stage 4: Text cleaning & normalisation
│   ├── codemix_analyzer.py          # Stage 4b: Code-mixing quantification
│   ├── terminology_mapper.py        # Stage 4c: Terminology mapping (colloquial → standard)
│   ├── concept_extract.py           # Stage 5: spaCy + KeyBERT concepts
│   ├── build_graph.py               # Stage 6: LLM prerequisite graph
│   ├── visualize_graph.py           # Stage 7: Per-video visualisations
│   ├── topological_sort.py          # Stage 8: Learning roadmap + flowchart
│   └── visualize_codemix.py         # Stage 9: Code-mix visualisations
│
├── data/
│   ├── video_links.txt              # Source YouTube URLs (one per line)
│   ├── videos/                      # Downloaded MP4 video files
│   ├── audios/                      # Extracted 16 kHz mono WAV audio
│   ├── transcripts/                 # Gemini transcriptions (code-mixed + English)
│   ├── cleaned_transcripts/         # Standardised segments (JSON)
│   ├── terminology_mappings/        # Per-video terminology mapping (JSON)
│   ├── concepts/                    # Extracted concepts with timestamps (JSON)
│   ├── graphs/                      # LLM-generated prerequisite graphs (JSON)
│   └── analysis/                    # Code-mixing analysis results (JSON)
│
└── outputs/
  └── <video>/                     # One directory per video
    ├── canonical.json           # Machine-readable graph
    ├── <video>.gexf             # GEXF graph export
    ├── interactive.html         # Interactive HTML graph
    ├── dependency.png           # Static dependency diagram
    ├── learning_roadmap.json    # Topologically sorted study plan
    ├── learning_roadmap.png     # Layered flowchart visualisation
    ├── codemix_profile.png      # Code-mix profile chart
    └── terminology_mapping.json # Terminology mapping copy for the video
```

---

