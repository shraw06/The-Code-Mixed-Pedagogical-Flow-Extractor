#!/usr/bin/env python3
"""
Generate a topologically sorted learning roadmap from a concept-dependency graph.

Given the prerequisite graph produced by build_graph.py (stored in
data/graphs/<video>.json), this module:

  1. Loads the graph as a NetworkX DiGraph.
  2. Breaks any cycles (using a lightweight feedback-arc-set heuristic) so
     the result is a true DAG.
  3. Performs a topological sort to produce a linear ordering of concepts
     where every prerequisite appears before the concepts that depend on it.
  4. Groups concepts into numbered *levels* — concepts at the same level
     can be studied in parallel because they share no mutual dependencies.
  5. Writes a JSON file (``learning_roadmap.json``) to the output directory.

The output JSON is structured as::

    {
      "video": "<stem>",
      "total_concepts": N,
      "levels": [
        {
          "level": 1,
          "concepts": [
            {
              "concept": "...",
              "prerequisites": [],
              "first_mentioned": 12.5,
              "frequency": 4,
              "explanation": "Why this comes here"
            },
            ...
          ]
        },
        ...
      ],
      "linear_order": ["concept_A", "concept_B", ...]
    }

Usage (standalone):
    python pipeline/topological_sort.py               # process all graphs
    python pipeline/topological_sort.py <graph.json>   # process one file
"""

import copy
import glob
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Re-use helpers from visualize_graph ──────────────────────────────────

def _parse_time_refs(refs):
    """Normalise time_references to list of (start, end) tuples."""
    out = []
    for r in refs:
        if isinstance(r, dict):
            out.append((float(r["start"]), float(r["end"])))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            out.append((float(r[0]), float(r[1])))
    return sorted(set(out))


def load_graph(path):
    """Return a NetworkX DiGraph from a graph JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()

    for n in data.get("nodes", []):
        name = n.get("concept", "").strip()
        if not name:
            continue
        times = _parse_time_refs(n.get("time_references", []))
        avg_t = np.mean([s for s, e in times]) if times else 0.0
        G.add_node(name, times=times, freq=len(times), avg_t=avg_t)

    for e in data.get("edges", []):
        src = e.get("source", "").strip()
        tgt = e.get("target", "").strip()
        if not src or not tgt:
            continue
        for nd in (src, tgt):
            if nd not in G:
                G.add_node(nd, times=[], freq=0, avg_t=0.0)
        expl = e.get("explanation", "")
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
            G[src][tgt]["explanation"] += "\n" + expl
        else:
            G.add_edge(src, tgt, weight=1, explanation=expl)

    return G


# ── Cycle breaking ───────────────────────────────────────────────────────

def _break_cycles(G: nx.DiGraph) -> nx.DiGraph:
    """Return a DAG copy of *G* by removing minimal feedback-arc-set edges.

    Strategy: iterate through cycles found by ``nx.simple_cycles`` and
    remove the edge with the lowest weight each time.  For most
    pedagogical graphs the number of cycles is small.
    """
    H = copy.deepcopy(G)
    while not nx.is_directed_acyclic_graph(H):
        try:
            cycle = next(nx.simple_cycles(H))
        except StopIteration:
            break
        # Find the weakest edge in the cycle
        weakest_edge = None
        min_weight = float("inf")
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if H.has_edge(u, v):
                w = H[u][v].get("weight", 1)
                if w < min_weight:
                    min_weight = w
                    weakest_edge = (u, v)
        if weakest_edge:
            H.remove_edge(*weakest_edge)
    return H


# ── Topological sort with level assignment ───────────────────────────────

def _assign_levels(G: nx.DiGraph) -> dict[str, int]:
    """BFS-based level assignment.  Level 0 = no prerequisites (roots)."""
    in_deg = dict(G.in_degree())
    levels: dict[str, int] = {}
    queue = [n for n, d in in_deg.items() if d == 0]
    current_level = 0
    while queue:
        for node in queue:
            levels[node] = current_level
        next_queue = []
        for node in queue:
            for succ in G.successors(node):
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    next_queue.append(succ)
        queue = next_queue
        current_level += 1
    # Handle any orphaned nodes (shouldn't happen after cycle breaking)
    for n in G.nodes():
        if n not in levels:
            levels[n] = current_level
    return levels


def _predecessors_for_node(G: nx.DiGraph, node: str) -> list[str]:
    """Direct prerequisites of *node* in the DAG."""
    return sorted(G.predecessors(node))


def _edge_explanation(G: nx.DiGraph, src: str, tgt: str) -> str:
    """Return the explanation string for edge src→tgt (if any)."""
    if G.has_edge(src, tgt):
        return (G[src][tgt].get("explanation", "") or "").strip()
    return ""


# ── Roadmap visualisation ────────────────────────────────────────────────

# Level-based colour palette (cool→warm as depth increases)
_LEVEL_COLOURS = [
    "#4e79a7",  # L1 – steel blue (foundations)
    "#59a14f",  # L2 – green
    "#f28e2b",  # L3 – orange
    "#e15759",  # L4 – coral
    "#b07aa1",  # L5 – purple
    "#edc948",  # L6 – gold
    "#76b7b2",  # L7 – teal
    "#ff9da7",  # L8 – pink
    "#9c755f",  # L9 – brown
    "#bab0ac",  # L10 – grey
]


def _wrap_label(text: str, max_chars: int = 18) -> str:
    """Wrap a concept label to fit inside a node box."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def make_roadmap_visual(roadmap: dict, dag: nx.DiGraph, out_path: str):
    """Render a layered learning-path flowchart from the roadmap JSON.

    Layout:
      - Each level is a horizontal row; level 1 at the top.
      - Nodes are rounded boxes coloured by level.
      - Node width scales slightly with label length; node height is fixed.
      - Prerequisite arrows flow top-to-bottom between levels.
      - A left-side label shows the level number.
    """
    levels_data = roadmap.get("levels", [])
    if not levels_data:
        return  # nothing to draw

    # ── geometry constants ───────────────────────────────────────────────
    ROW_HEIGHT = 1.6          # vertical space per level
    NODE_H = 0.72             # box height
    NODE_PAD_X = 0.35         # extra horizontal padding per character
    MIN_NODE_W = 2.2          # minimum box width
    COL_GAP = 0.6             # gap between nodes in same level
    MARGIN_LEFT = 1.6         # space for level labels
    MARGIN_RIGHT = 0.6
    MARGIN_TOP = 1.0
    MARGIN_BOT = 0.5
    FONT_SIZE = 8.5

    # ── compute node positions ───────────────────────────────────────────
    # pos[concept] = (cx, cy)    box_w[concept] = width
    pos: dict[str, tuple[float, float]] = {}
    box_w: dict[str, float] = {}
    max_row_w = 0.0
    num_levels = len(levels_data)

    for lvl_data in levels_data:
        lvl_idx = lvl_data["level"] - 1  # 0-based
        concepts = lvl_data["concepts"]
        # Compute widths for each node in this row
        widths = []
        for c in concepts:
            label = _wrap_label(c["concept"])
            max_line = max((len(l) for l in label.split("\n")), default=0)
            w = max(MIN_NODE_W, max_line * 0.13 + NODE_PAD_X * 2)
            widths.append(w)
            box_w[c["concept"]] = w
        total_w = sum(widths) + COL_GAP * max(0, len(widths) - 1)
        max_row_w = max(max_row_w, total_w)

    # Centre each row horizontally
    fig_w = MARGIN_LEFT + max_row_w + MARGIN_RIGHT
    fig_h = MARGIN_TOP + num_levels * ROW_HEIGHT + MARGIN_BOT

    for lvl_data in levels_data:
        lvl_idx = lvl_data["level"] - 1
        concepts = lvl_data["concepts"]
        widths = [box_w[c["concept"]] for c in concepts]
        total_w = sum(widths) + COL_GAP * max(0, len(widths) - 1)
        start_x = MARGIN_LEFT + (max_row_w - total_w) / 2
        cx = start_x
        cy = MARGIN_TOP + lvl_idx * ROW_HEIGHT + NODE_H / 2
        for i, c in enumerate(concepts):
            w = widths[i]
            pos[c["concept"]] = (cx + w / 2, cy)
            cx += w + COL_GAP

    # ── draw ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(fig_w, 6), max(fig_h, 4)), dpi=180)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)  # y increases downward
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    title = roadmap.get("video", "")
    short_title = (title[:60] + "…") if len(title) > 60 else title
    ax.text(fig_w / 2, 0.35, f"Learning Roadmap — {short_title}",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="#24292e")

    # Level labels
    for lvl_data in levels_data:
        lvl_idx = lvl_data["level"] - 1
        cy = MARGIN_TOP + lvl_idx * ROW_HEIGHT + NODE_H / 2
        ax.text(MARGIN_LEFT * 0.45, cy, f"Level {lvl_data['level']}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#555", style="italic")

    # Prerequisite arrows (draw before boxes so they go behind)
    for lvl_data in levels_data:
        for c in lvl_data["concepts"]:
            cname = c["concept"]
            cx, cy = pos[cname]
            for prereq in c["prerequisites"]:
                if prereq not in pos:
                    continue
                px, py = pos[prereq]
                # Arrow from bottom of prereq box to top of current box
                ax.annotate(
                    "",
                    xy=(cx, cy - NODE_H / 2),          # arrow head (top of child)
                    xytext=(px, py + NODE_H / 2),       # arrow tail (bottom of prereq)
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#aaaaaa",
                        lw=1.0,
                        connectionstyle="arc3,rad=0.08",
                        shrinkA=2, shrinkB=2,
                    ),
                )

    # Node boxes
    for lvl_data in levels_data:
        lvl_idx = lvl_data["level"] - 1
        colour = _LEVEL_COLOURS[lvl_idx % len(_LEVEL_COLOURS)]
        # Lighter fill
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(colour)
        fill = tuple(min(1.0, c * 0.3 + 0.7) for c in rgb)

        for c in lvl_data["concepts"]:
            cname = c["concept"]
            cx, cy = pos[cname]
            w = box_w[cname]
            rect = FancyBboxPatch(
                (cx - w / 2, cy - NODE_H / 2), w, NODE_H,
                boxstyle="round,pad=0.08",
                facecolor=fill,
                edgecolor=colour,
                linewidth=1.8,
            )
            ax.add_patch(rect)
            label = _wrap_label(cname)
            n_lines = label.count("\n") + 1
            ax.text(cx, cy, label,
                    ha="center", va="center",
                    fontsize=FONT_SIZE - (0.5 if n_lines > 2 else 0),
                    fontweight="600", color="#24292e",
                    linespacing=1.15)

            # Small frequency badge (top-right corner)
            freq = c.get("frequency", 0)
            if freq > 0:
                bx = cx + w / 2 - 0.15
                by = cy - NODE_H / 2 + 0.10
                ax.text(bx, by, f"×{freq}", ha="right", va="bottom",
                        fontsize=5.5, color=colour, fontweight="bold")

    # Legend
    handles = []
    seen = set()
    for lvl_data in levels_data:
        idx = lvl_data["level"] - 1
        if idx not in seen:
            seen.add(idx)
            c = _LEVEL_COLOURS[idx % len(_LEVEL_COLOURS)]
            handles.append(mpatches.Patch(color=c, label=f"Level {lvl_data['level']}"))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7,
                  framealpha=0.85, edgecolor="#ddd",
                  bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight",
                facecolor="#fdfdfd", edgecolor="none")
    plt.close(fig)


# ── Public API ───────────────────────────────────────────────────────────

def generate_roadmap(graph_json_path: str) -> dict:
    """Build a topologically-sorted learning roadmap from a graph JSON.

    Returns the roadmap dict (also written to disk).
    """
    graph_json_path = Path(graph_json_path)
    stem = graph_json_path.stem

    G = load_graph(str(graph_json_path))

    # Ensure we have a DAG
    dag = _break_cycles(G) if not nx.is_directed_acyclic_graph(G) else G

    # Topological order (Kahn's algorithm flavour that respects avg_t for
    # tie-breaking so concepts mentioned earlier in the video appear first)
    topo_order = list(
        nx.lexicographical_topological_sort(
            dag, key=lambda n: dag.nodes[n].get("avg_t", 0.0)
        )
    )

    levels = _assign_levels(dag)

    # Group by level
    max_level = max(levels.values()) if levels else 0
    level_groups: list[dict] = []
    for lvl in range(max_level + 1):
        members = [n for n in topo_order if levels.get(n) == lvl]
        if not members:
            continue
        concepts = []
        for c in members:
            d = dag.nodes[c]
            times = d.get("times", [])
            first_mentioned = min((s for s, e in times), default=None)
            prereqs = _predecessors_for_node(dag, c)
            # Collect explanations of why each prerequisite feeds into this concept
            why_parts = []
            for p in prereqs:
                expl = _edge_explanation(dag, p, c)
                if expl:
                    why_parts.append(expl)
            concepts.append({
                "concept": c,
                "prerequisites": prereqs,
                "first_mentioned": round(first_mentioned, 2) if first_mentioned is not None else None,
                "frequency": int(d.get("freq", 0)),
                "explanation": " | ".join(why_parts) if why_parts else (
                    "Foundational concept — no prerequisites."
                    if not prereqs else ""
                ),
            })
        level_groups.append({
            "level": lvl + 1,  # 1-indexed for human readability
            "concepts": concepts,
        })

    roadmap = {
        "video": stem,
        "total_concepts": len(topo_order),
        "levels": level_groups,
        "linear_order": topo_order,
    }

    # Write output
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "learning_roadmap.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(roadmap, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Learning roadmap  → {out_path}  ({len(topo_order)} concepts, {len(level_groups)} levels)")

    # Generate visual flowchart
    png_path = out_dir / "learning_roadmap.png"
    try:
        make_roadmap_visual(roadmap, dag, str(png_path))
        print(f"  ✓ Roadmap visual    → {png_path}")
    except Exception as exc:
        print(f"  ⚠ Roadmap visual skipped: {exc}")

    return roadmap


# ── CLI entry point ──────────────────────────────────────────────────────

def process_all():
    """Process every graph JSON found in data/graphs/."""
    graphs_dir = Path(__file__).resolve().parent.parent / "data" / "graphs"
    graph_files = sorted(glob.glob(str(graphs_dir / "*.json")))
    if not graph_files:
        print("No graph JSON files found in data/graphs/")
        return
    for gf in graph_files:
        stem = Path(gf).stem
        print(f"Processing: {stem}")
        generate_roadmap(gf)
    print("\nDone. Roadmaps are in outputs/<video>/learning_roadmap.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            generate_roadmap(arg)
    else:
        process_all()
