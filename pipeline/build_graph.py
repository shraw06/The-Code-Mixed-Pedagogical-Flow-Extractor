import json
import glob
import os
from pathlib import Path
from time import sleep

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    pass

# Try to load environment variables from a .env file in the project root (parent dir)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(env_path)
except Exception:
    # If python-dotenv is not installed or .env is missing, fall back to existing env vars.
    # Installing python-dotenv is recommended for local development:
    # pip install python-dotenv
    pass

def generate_graph(concepts_file):
    with open(concepts_file, 'r') as f:
        data = json.load(f)
    
    # include the video title (derived from the concepts filename) in the prompt
    video_title = Path(concepts_file).stem
    # make the title human-friendly
    video_title = video_title.replace('_', ' ').strip()

    # ── Detect language metadata from the cleaned transcript ──
    # The cleaned transcript JSON preserves the "language" field from STT.
    detected_language = "code-mixed (unknown)"
    cleaned_transcript_path = Path(concepts_file).parent.parent / "cleaned_transcripts" / (Path(concepts_file).stem + ".json")
    if cleaned_transcript_path.exists():
        try:
            with open(cleaned_transcript_path, "r", encoding="utf-8") as ctf:
                ct_data = json.load(ctf)
            # Collect unique language labels from segments
            langs = set()
            for seg in ct_data:
                lang = seg.get("language", "")
                if lang and lang != "unknown":
                    langs.add(lang)
            if langs:
                detected_language = ", ".join(sorted(langs))
        except Exception:
            pass

    prompt = f"Video title: {video_title}\n"
    prompt += f"Detected code-mixed language(s): {detected_language}\n\n"
    prompt += "I have extracted the following candidate concepts from a code-mixed educational video transcript, along with the time segments (start and end in seconds) where each concept is discussed:\n"
    prompt += json.dumps(data, indent=2)
    prompt += "\n\nIMPORTANT CONTEXT: This transcript was produced from a code-mixed lecture where the teacher speaks in "
    prompt += f"{detected_language}. Technical terms are typically spoken in English while explanations are in the regional language. "
    prompt += "The concepts below have been extracted from the English-normalised version of the transcript.\n\n"
    prompt += "INSTRUCTIONS (important):\n"
    prompt += "1) REFINE CONCEPTS: Remove generic, off-topic, or redundant concepts. Merge synonyms and near-duplicates into a single canonical concept. Keep only concepts that are pedagogically relevant to the video topic.\n"
    prompt += "2) PEDAGOGICAL GRAPH (DAG): Build a directed graph representing the teacher's pedagogical flow. Edges must point from prerequisite -> dependent. Prefer edges where the source concept is introduced earlier in time than the target. If cycles are present, break them by removing the weakest/least-evidenced edges so the final graph is acyclic.\n"
    prompt += "3) OUTPUT SCHEMA (strict JSON ONLY): Produce a single JSON object with two keys: 'nodes' (array) and 'edges' (array). No explanatory text.\n"
    prompt += "   - Node object: {\"concept\": string, \"time_references\": [[start,end], ...], \"confidence\": float (0-1), \"order\": integer (pedagogical order, 0 = first)}\n"
    prompt += "   - Edge object: {\"source\": concept_string, \"target\": concept_string, \"explanation\": string, \"confidence\": float (0-1)}\n"
    prompt += "4) ORDERING: Assign an 'order' integer to each node approximating teaching sequence (lower = earlier).\n"
    prompt += "5) NO EXTRA TEXT: Return ONLY the JSON object (no markdown, no code fences, no commentary). Ensure valid JSON.\n\n"
    prompt += "Produce the refined, pedagogically-ordered concept graph now, following the schema exactly."
    
    try:
        if not os.environ.get("GEMINI_API_KEY"):
            print("Set GEMINI_API_KEY to run the Graph generation for", concepts_file)
            return

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        # Defensive extraction of raw text from response
        raw = ""
        if hasattr(response, "text") and response.text is not None:
            raw = response.text
        elif hasattr(response, "content") and response.content is not None:
            # content might be bytes
            raw = response.content.decode("utf-8", errors="ignore") if isinstance(response.content, (bytes, bytearray)) else str(response.content)
        elif hasattr(response, "data"):
            try:
                raw = json.dumps(response.data)
            except Exception:
                raw = str(response.data)
        else:
            raw = str(response)

        # Save raw response for debugging
        graph_dir = "data/graphs"
        os.makedirs(graph_dir, exist_ok=True)
        raw_debug_file = os.path.join(graph_dir, Path(concepts_file).stem + ".raw.txt")
        with open(raw_debug_file, "w", encoding="utf-8") as rf:
            rf.write(raw)

        if not raw.strip():
            print(f"Empty response from LLM for {concepts_file}. Raw saved to {raw_debug_file}")
            return

        def extract_json_from_text(text):
            """Attempt to extract a JSON object/array from a larger text blob.

            Strategy:
            - Try json.loads on full text
            - Find balanced {...} and [...] substrings and attempt to parse them
            - For each candidate, try light repairs (smart quotes, remove trailing commas, strip JS comments)
            - Return the first successfully parsed object or None
            """
            import re
            # 1) direct
            try:
                return json.loads(text)
            except Exception:
                pass

            candidates = []

            # 2) find balanced {...} and [...] substrings
            opens = {'{': '}', '[': ']'}
            for i, ch in enumerate(text):
                if ch in opens:
                    openc = ch
                    closec = opens[ch]
                    depth = 0
                    for j in range(i, len(text)):
                        if text[j] == openc:
                            depth += 1
                        elif text[j] == closec:
                            depth -= 1
                            if depth == 0:
                                candidates.append(text[i:j+1])
                                break

            # 3) attempt to parse candidates, with light repairs
            for s in candidates:
                try:
                    return json.loads(s)
                except Exception:
                    # try light fixes
                    s2 = s
                    # normalize smart quotes
                    s2 = s2.replace('\u201c', '"').replace('\u201d', '"')
                    s2 = s2.replace('\u2018', "'").replace('\u2019', "'")
                    # remove JS/C-style comments
                    s2 = re.sub(r'//.*?\n', '\n', s2)
                    s2 = re.sub(r'/\*[\s\S]*?\*/', '', s2)
                    # remove trailing commas before } or ]
                    s2 = re.sub(r',\s*(?=[}\]])', '', s2)
                    try:
                        return json.loads(s2)
                    except Exception:
                        continue

            return None

        output_data = extract_json_from_text(raw)
        if output_data is None:
            print(f"JSON extraction failed for {concepts_file}. Raw saved to {raw_debug_file}")
            return

        graph_file = concepts_file.replace("data/concepts/", graph_dir + "/")
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
            
        print(f"Graph saved to {graph_file} (raw response saved to {raw_debug_file})")

    except Exception as e:
        print(f"Error calling LLM for {concepts_file}: {e}")

if __name__ == "__main__":
    for concepts_file in glob.glob("data/concepts/*.json"):
        generate_graph(concepts_file)
        sleep(10)
