#!/usr/bin/env python3
"""
Step 7c – Code-Mixing & Cross-Video Visualisations.

Generates additional visualisations that highlight the code-mixed nature
of the data and cross-video concept relationships:

  Per video (in outputs/<video>/):
    • codemix_profile.png    — language distribution pie + CMI timeline

  Cross-video (in outputs/cross_video/):
    • similarity_heatmap.png — Jaccard concept-similarity matrix
    • bridging_network.png   — bridging concepts highlighted across videos
    • codemix_comparison.png — bar chart comparing CMI across videos

Usage:
    python pipeline/visualize_codemix.py
"""

import json
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "data" / "analysis"
OUTPUTS_DIR = ROOT / "outputs"
CROSS_DIR = OUTPUTS_DIR / "cross_video"
CROSS_DIR.mkdir(parents=True, exist_ok=True)


_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

_SCRIPT_COLOURS = {
    "LATIN": "#4e79a7",
    "DEVANAGARI": "#f28e2b",
    "TAMIL": "#e15759",
    "TELUGU": "#76b7b2",
    "KANNADA": "#59a14f",
    "MALAYALAM": "#edc948",
    "BENGALI": "#b07aa1",
    "OTHER": "#bab0ac",
}


def _short(name: str, maxlen: int = 35) -> str:
    return name[:maxlen - 3] + "..." if len(name) > maxlen else name


# ─────────────────────────────────────────────
# Per-video: Code-mix profile
# ─────────────────────────────────────────────

def make_codemix_profile(analysis_data: dict, out_path: str):
    """Two-panel figure: language distribution pie + CMI over time."""
    segs = analysis_data.get("segment_analyses", [])
    if not segs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("#fafafa")

    # ── Left: Script distribution pie chart ──
    script_dist = analysis_data.get("script_distribution", {})
    if script_dist:
        labels = list(script_dist.keys())
        values = list(script_dist.values())
        colours = [_SCRIPT_COLOURS.get(l, "#ccc") for l in labels]
        wedges, texts, autotexts = ax1.pie(
            values, labels=labels, colors=colours,
            autopct="%1.1f%%", startangle=140,
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")
    else:
        ax1.text(0.5, 0.5, "No script data", ha="center", va="center")

    lang_labels = analysis_data.get("language_label_distribution", {})
    lang_str = ", ".join(f"{k}" for k in lang_labels.keys())
    ax1.set_title(f"Script Distribution\n(Gemini labels: {lang_str})", fontsize=10, fontweight="bold")

    # Add CMI annotation
    global_cmi = analysis_data.get("global_cmi", 0)
    rom_frac = analysis_data.get("romanised_segment_fraction", 0)
    ax1.annotate(
        f"Global CMI: {global_cmi:.3f}\nRomanised: {rom_frac:.0%}",
        xy=(0.5, -0.08), xycoords="axes fraction",
        ha="center", fontsize=9, fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffacd", alpha=0.8),
    )

    # ── Right: CMI timeline (segment-level) ──
    times = [(s["start"] + s["end"]) / 2 for s in segs]
    cmis = [s["cmi"] for s in segs]
    switches = [s["n_switches"] for s in segs]

    ax2.bar(times, cmis, width=max(1, (max(times) - min(times)) / len(times) * 0.8),
            color="#4e79a7", alpha=0.7, label="Segment CMI")
    ax2.set_xlabel("Time (seconds)", fontsize=9)
    ax2.set_ylabel("Code-Mixing Index", fontsize=9, color="#4e79a7")
    ax2.tick_params(axis="y", labelcolor="#4e79a7")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Code-Mixing Over Time", fontsize=10, fontweight="bold")

    # Overlay switch count on twin axis
    ax3 = ax2.twinx()
    ax3.plot(times, switches, color="#e15759", linewidth=1.2, alpha=0.7, label="Switches")
    ax3.scatter(times, switches, color="#e15759", s=10, alpha=0.5, zorder=5)
    ax3.set_ylabel("Language Switches", fontsize=9, color="#e15759")
    ax3.tick_params(axis="y", labelcolor="#e15759")

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle(_short(analysis_data.get("video", ""), 80), fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Per-video: Language switch heatmap
# ─────────────────────────────────────────────

def make_switch_heatmap(analysis_data: dict, out_path: str):
    """Horizontal heatmap showing language composition per segment over time."""
    segs = analysis_data.get("segment_analyses", [])
    if not segs:
        return

    # Collect all scripts seen
    all_scripts = set()
    for s in segs:
        all_scripts.update(s.get("script_counts", {}).keys())
    all_scripts = sorted(all_scripts)
    if not all_scripts:
        return

    # Build matrix: rows = scripts, cols = segments (time-ordered)
    matrix = np.zeros((len(all_scripts), len(segs)))
    for j, seg in enumerate(segs):
        sc = seg.get("script_counts", {})
        total = sum(sc.values()) or 1
        for i, script in enumerate(all_scripts):
            matrix[i, j] = sc.get(script, 0) / total

    fig, ax = plt.subplots(figsize=(max(12, len(segs) * 0.15), max(3, len(all_scripts) * 0.8)))
    fig.patch.set_facecolor("#fafafa")

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(all_scripts)))
    ax.set_yticklabels(all_scripts, fontsize=9)
    ax.set_xlabel("Segment index (time →)", fontsize=9)
    ax.set_title(f"Language Composition Per Segment\n{_short(analysis_data.get('video', ''), 70)}", fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Word fraction", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Cross-video: CMI comparison bar chart
# ─────────────────────────────────────────────

def make_cmi_comparison(analysis_files: list[str], out_path: str):
    """Bar chart comparing Global CMI + switch density across all videos."""
    data_items = []
    for fp in sorted(analysis_files):
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        data_items.append(d)

    if not data_items:
        return

    names = [_short(d["video"], 30) for d in data_items]
    cmis = [d.get("global_cmi", 0) for d in data_items]
    switch_dens = [d.get("switch_density_per_100_words", 0) for d in data_items]
    rom_fracs = [d.get("romanised_segment_fraction", 0) for d in data_items]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2.5), 6))
    fig.patch.set_facecolor("#fafafa")

    bars1 = ax.bar(x - width, cmis, width, label="Global CMI", color="#4e79a7", alpha=0.85)
    bars2 = ax.bar(x, [s / 100 for s in switch_dens], width,
                   label="Switch Density / 100", color="#e15759", alpha=0.85)
    bars3 = ax.bar(x + width, rom_fracs, width, label="Romanised Fraction", color="#f28e2b", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score (0–1)", fontsize=10)
    ax.set_title("Code-Mixing Comparison Across Videos", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 1.15)

    # Annotate bar values
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def process_all():
    # ── Per-video code-mix visualisations ──
    codemix_files = sorted(glob.glob(str(ANALYSIS_DIR / "*_codemix.json")))
    if not codemix_files:
        print("No code-mix analysis files found. Run codemix_analyzer.py first.")
        return

    for fp in codemix_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        video = data.get("video", Path(fp).stem.replace("_codemix", ""))
        out_dir = OUTPUTS_DIR / video
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Visualising code-mix: {_short(video, 50)}")

        make_codemix_profile(data, str(out_dir / "codemix_profile.png"))
        print(f"  ✓ codemix_profile.png")

    # ── Cross-video CMI comparison ──
    print("\nCross-video visualisations:")

    make_cmi_comparison(codemix_files, str(CROSS_DIR / "codemix_comparison.png"))
    print(f"  ✓ codemix_comparison.png")

    print("\nDone. Visualisations are in outputs/*/")


if __name__ == "__main__":
    process_all()
