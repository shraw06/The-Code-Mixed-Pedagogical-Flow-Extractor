#!/usr/bin/env python3
"""
Visualize concept-interdependency graphs produced by build_graph.py.

For each JSON in data/graphs/*.json this script creates outputs
inside outputs/<video-stem>/:

  1. interactive.html  – force-directed interactive graph  (pyvis)
  2. dependency.png    – hierarchical static dependency graph (matplotlib)
  3. canonical.json    – strictly validated machine-readable graph
  4. <stem>.gexf       – standard graph exchange format (Gephi, igraph)

Usage:
    pip install networkx matplotlib pyvis numpy
    python pipeline/visualize_graph.py
"""

import json
import glob
import os
import textwrap
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from pyvis.network import Network


# Helpers

def _parse_time_refs(refs):
    """Normalise time_references to list of (start, end) tuples.
       Handles both [{start, end}] dicts and [[start, end]] arrays."""
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
        # auto-add missing nodes (LLM might mention concepts not in nodes list)
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


# 1. Interactive HTML (pyvis)

# Colour palette – 12 distinct hues
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac", "#d37295", "#fabfd2",
]


def _node_colour(avg_t, max_t):
    """Map avg timestamp to a warm-to-cool colour gradient."""
    if max_t == 0:
        return "#4e79a7"
    ratio = min(avg_t / max_t, 1.0)
    # blue → red gradient via HSV
    import colorsys
    h = 0.6 - 0.6 * ratio  # 0.6 (blue) → 0.0 (red)
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.92)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))


def make_interactive(G, title, out_path):
    """Produce an interactive HTML visualisation."""
    net = Network(
        height="900px", width="100%",
        directed=True, notebook=False,
        bgcolor="#fdfdfd", font_color="#333",
    )
    net.heading = title
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 180,
          "nodeSpacing": 120,
          "treeSpacing": 200,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": false,
          "direction": "LR"
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      },
      "physics": {
        "enabled": false
      }
    }
    """)
    # net.barnes_hut(gravity=-8000, central_gravity=0.35,
    #                spring_length=160, spring_strength=0.02)

    max_t = max((d.get("avg_t", 0) for _, d in G.nodes(data=True)), default=1.0) or 1.0
    # max_freq = max((d.get("freq", 1) for _, d in G.nodes(data=True)), default=1)

    for n, d in G.nodes(data=True):
        freq = d.get("freq", 1)
        avg_t = d.get("avg_t", 0)
        times = d.get("times", [])
        times_str = ", ".join(f"{s:.0f}-{e:.0f}s" for s, e in times[:8])
        if len(times) > 8:
            times_str += f" … (+{len(times)-8} more)"
        tooltip = f"<b>{n}</b><br>Mentions: {freq}<br>Avg time: {avg_t:.1f}s<br>Segments: {times_str}"
        # size = 12 + freq * 3
        colour = _node_colour(avg_t, max_t)
        net.add_node(n, label=n, title=tooltip, color=colour,
                     font={"size": 14, "face": "Arial"})

    for u, v, d in G.edges(data=True):
        expl = d.get("explanation", "")
        wrapped = "<br>".join(textwrap.wrap(expl, 60)) if expl else ""
        net.add_edge(u, v, title=wrapped, value=d.get("weight", 1),
                     arrows="to", color={"color": "#888", "opacity": 0.6})

    # net.set_options("""
    # {
    #   "interaction": {
    #     "hover": true,
    #     "tooltipDelay": 100
    #   },
    #   "physics": {
    #     "stabilization": {"iterations": 200}
    #   }
    # }
    # """)
    net.save_graph(str(out_path))

    # Post-process HTML to ensure the title/heading appears only once. Pyvis may
    # render the heading in more than one place depending on template/version.
    try:
        import re
        p = Path(out_path)
        html = p.read_text(encoding="utf-8")
        # match <h1 ...>title</h1> (allow attributes and whitespace)
        h1_re = re.compile(r"(<h1[^>]*>\s*" + re.escape(title) + r"\s*</h1>)", re.IGNORECASE)
        matches = list(h1_re.finditer(html))
        if len(matches) > 1:
            # remove all but the first occurrence (iterate reversed to preserve indices)
            for m in reversed(matches[1:]):
                html = html[:m.start()] + html[m.end():]
            p.write_text(html, encoding="utf-8")
    except Exception:
        # non-fatal — leave the generated HTML as-is if post-processing fails
        pass


# 2. Hierarchical static PNG (matplotlib)

def _hierarchy_pos(G):
    """Position nodes by topological layers (Y) and spread within layer (X).
    Attempt Graphviz 'dot' (hierarchical) layout first; fallback to existing method.
    """
    # try graphviz via networkx wrappers
    try:
        # prefer nx_pydot if available
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
            # graphviz gives positions in large coords; normalize spacing
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            width = max(1.0, maxx - minx)
            height = max(1.0, maxy - miny)
            norm = {n: ((x - minx) / width * len(G), -(y - miny) / height * (len(G)/2)) for n, (x, y) in pos.items()}
            return norm
        except Exception:
            # try agraph (pygraphviz) wrapper
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                pos = graphviz_layout(G, prog="dot")
                xs = [p[0] for p in pos.values()]
                ys = [p[1] for p in pos.values()]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                width = max(1.0, maxx - minx)
                height = max(1.0, maxy - miny)
                norm = {n: ((x - minx) / width * len(G), -(y - miny) / height * (len(G)/2)) for n, (x, y) in pos.items()}
                return norm
            except Exception:
                raise
    except Exception:
        # fallback to original layered layout
        try:
            layers = list(nx.topological_generations(G))
        except nx.NetworkXUnfeasible:
            # graph has cycles – fall back to spring
            return nx.spring_layout(G, seed=42, k=2.5)

        pos = {}
        for depth, layer in enumerate(layers):
            layer = sorted(layer)
            n = len(layer)
            for i, node in enumerate(layer):
                x = (i - n / 2) * 1.6
                y = -depth * 1.8
                pos[node] = (x, y)
        return pos


def make_static(G, title, out_path):
    """Produce a layered dependency PNG."""
    if len(G) == 0:
        return

    pos = _hierarchy_pos(G)

    max_t = max((d.get("avg_t", 0) for _, d in G.nodes(data=True)), default=1.0) or 1.0
    node_colours = [_node_colour(G.nodes[n].get("avg_t", 0), max_t) for n in G.nodes()]
    # node_sizes = [200 + G.nodes[n].get("freq", 1) * 60 for n in G.nodes()]

    widths = [1.0 + 0.9 * d.get("weight", 1) for _, _, d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(max(16, len(G) * 0.6), max(10, len(G) * 0.35)))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    xmap = {n: pos[n][0] for n in pos}
    edge_rads = []
    for u, v in G.edges():
        dx = abs(xmap.get(u, 0) - xmap.get(v, 0))
        edge_rads.append(0.02 + 0.08 * (dx / (max(1.0, max(abs(x) for x in xmap.values())))))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#aaa", width=widths,
        arrows=True, arrowsize=14, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08", alpha=0.8,
        # min_source_margin=12, min_target_margin=12,
    )

    # nx.draw_networkx_nodes(
    #     G, pos, ax=ax,
    #     node_color=node_colours, node_size=node_sizes,
    #     edgecolors="#555", linewidths=0.8,
    # )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colours, node_size=300,
        edgecolors="#555", linewidths=0.8,
    )

    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8, font_weight="bold", font_family="sans-serif",
    )

    # legend: colour gradient = timeline position
    import matplotlib.colors as mcolors
    import colorsys
    gradient = []
    for i in range(256):
        h = 0.6 - 0.6 * (i / 255)
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.92)
        gradient.append((r, g, b))
    cmap = mcolors.ListedColormap(gradient)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_t))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Avg. appearance time (seconds)", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# 3. Timeline heatmap with dependency arrows
# ─────────────────────────────────────────────

def make_timeline(G, title, out_path):
    """Concept-timeline heatmap: concepts on Y, time on X, coloured bars."""
    if len(G) == 0:
        return

    # Sort concepts by first appearance
    concepts = sorted(
        G.nodes(),
        key=lambda n: min((s for s, e in G.nodes[n].get("times", [(0, 0)])), default=0),
    )

    # Deduplicate: keep only concepts that have time references
    concepts = [c for c in concepts if G.nodes[c].get("times")]
    if not concepts:
        return

    concept_idx = {c: i for i, c in enumerate(concepts)}
    max_time = max(e for c in concepts for s, e in G.nodes[c]["times"])

    fig, ax = plt.subplots(figsize=(max(14, max_time / 20), max(6, len(concepts) * 0.38)))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#f5f5f5")

    # colour per concept (cycling palette)
    cpal = _PALETTE * ((len(concepts) // len(_PALETTE)) + 1)

    bar_height = 0.6
    for ci, concept in enumerate(concepts):
        colour = cpal[ci]
        for s, e in G.nodes[concept]["times"]:
            width = max(e - s, 0.5)  # ensure visible
            rect = mpatches.FancyBboxPatch(
                (s, ci - bar_height / 2), width, bar_height,
                boxstyle="round,pad=0.1",
                facecolor=colour, edgecolor="white", linewidth=0.5, alpha=0.8,
            )
            ax.add_patch(rect)

    # overlay dependency arrows (source → target)
    drawn_edges = set()
    for u, v, d in G.edges(data=True):
        if u not in concept_idx or v not in concept_idx:
            continue
        key = (u, v)
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        # arrow from end of source's last segment to start of target's first segment
        u_times = G.nodes[u]["times"]
        v_times = G.nodes[v]["times"]
        x_start = max(e for s, e in u_times)
        x_end = min(s for s, e in v_times)
        y_start = concept_idx[u]
        y_end = concept_idx[v]
        ax.annotate(
            "", xy=(x_end, y_end), xytext=(x_start, y_start),
            arrowprops=dict(arrowstyle="-|>", color="#666", lw=0.8, alpha=0.45,
                            connectionstyle="arc3,rad=0.15"),
        )

    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts, fontsize=7, fontfamily="sans-serif")
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_xlim(-1, max_time + 5)
    ax.set_ylim(-0.8, len(concepts) - 0.2)
    ax.invert_yaxis()
    ax.set_title(f"{title}\nConcept Timeline & Dependencies", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

import copy


# Canonical machine-readable export


def export_canonical_graph(G, out_dir, stem):
    """
    Write a strictly validated, machine-readable canonical JSON and GEXF
    for the given NetworkX DiGraph. Returns the path to canonical.json.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = []
    for n, d in sorted(G.nodes(data=True), key=lambda x: x[1].get("avg_t", 0)):
        times = sorted(((float(s), float(e)) for s, e in d.get("times", [])), key=lambda x: x[0])
        node_entry = {
            "id": n,
            "concept": n,
            "start": times[0][0] if times else None,
            "end": max(e for _, e in times) if times else None,
            "avg_t": round(float(d.get("avg_t", 0.0)), 3),
            "freq": int(d.get("freq", 0)),
            "time_references": [{"start": s, "end": e} for s, e in times],
        }
        nodes.append(node_entry)

    edges = []
    node_ids = {n["id"] for n in nodes}
    for u, v, d in G.edges(data=True):
        # Only emit edges where both endpoints exist in nodes list
        if u not in node_ids or v not in node_ids:
            continue
        edges.append({
            "source": u,
            "target": v,
            "weight": int(d.get("weight", 1)),
            "explanation": (d.get("explanation", "") or "").strip(),
        })

    canonical = {
        "video": stem,
        "nodes": nodes,
        "edges": edges,
    }

    # Validate: every edge source/target must exist in nodes
    ids = {n["id"] for n in nodes}
    bad_edges = [e for e in edges if e["source"] not in ids or e["target"] not in ids]
    if bad_edges:
        print(f"  ⚠ {len(bad_edges)} edges reference missing nodes — they were excluded.")

    can_path = out_dir / "canonical.json"
    with open(can_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)

    # GEXF — standard graph exchange format (Gephi, igraph, etc.)
    # networkx.write_gexf requires all edge/node attributes to be
    # primitive types; coerce here to avoid unpacking errors.
    try:
        G_gexf = nx.DiGraph()
        for n, d in G.nodes(data=True):
            G_gexf.add_node(n, freq=int(d.get("freq", 0)), avg_t=float(d.get("avg_t", 0.0)))
        for u, v, d in G.edges(data=True):
            G_gexf.add_edge(u, v, weight=float(d.get("weight", 1)),
                            explanation=str(d.get("explanation", "") or ""))
        nx.write_gexf(G_gexf, str(out_dir / f"{stem}.gexf"))
    except Exception as ex:
        print(f"  ⚠ GEXF export skipped: {ex}")

    return can_path


def simplify_graph(G, method="auto"):
    """
    Return a simplified copy of G with redundant direct edges removed.

    - method="auto": use transitive_reduction() if G is a DAG, else use heuristic.
    - method="transitive_reduction": require DAG (raises if not).
    - method="heuristic": remove an edge (u,v) if there exists an alternate path u->v
      after temporarily removing the edge. Edges are considered from lowest weight first.
    """
    if method == "auto":
        method = "transitive_reduction" if nx.is_directed_acyclic_graph(G) else "heuristic"

    Gs = copy.deepcopy(G)

    if method == "transitive_reduction":
        try:
            # networkx.transitive_reduction returns a new DiGraph without attributes,
            # so we reconstruct attributes from G.
            TR = nx.transitive_reduction(Gs)
            newG = nx.DiGraph()
            for n, d in Gs.nodes(data=True):
                newG.add_node(n, **d)
            for u, v in TR.edges():
                # copy attributes from original edge if present
                if Gs.has_edge(u, v):
                    newG.add_edge(u, v, **Gs[u][v])
                else:
                    newG.add_edge(u, v, weight=1, explanation="")
            return newG
        except Exception:
            # fall back to heuristic if something goes wrong
            method = "heuristic"

    if method == "heuristic":
        # Sort edges by increasing weight (we prefer to remove weaker edges first)
        edges = sorted(Gs.edges(data=True), key=lambda x: x[2].get("weight", 1))
        # operate on a mutable graph copy
        H = Gs
        for u, v, attr in edges:
            if not H.has_edge(u, v):
                continue
            # temporarily remove edge and test reachability
            saved = H.remove_edge(u, v)
            # use has_path on directed graph
            try:
                reachable = nx.has_path(H, u, v)
            except Exception:
                reachable = False
            if reachable:
                # permanent removal (edge already removed)
                continue
            else:
                # restore edge (no alternative path)
                H.add_edge(u, v, **attr)
        return H

    return Gs
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def process_graph(json_path):
    stem = Path(json_path).stem
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {stem}")
    G = load_graph(json_path)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    G = simplify_graph(G, method="auto")
    print(f"  Simplified: Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Export strict canonical machine-readable representation
    can_path = export_canonical_graph(G, out_dir, stem)
    print(f"  ✓ Canonical JSON    → {can_path}")
    print(f"  ✓ GEXF              → {out_dir / (stem + '.gexf')}")

    # 1 – Interactive HTML
    html_path = out_dir / "interactive.html"
    make_interactive(G, stem, html_path)
    print(f"  ✓ Interactive graph → {html_path}")

    # 2 – Static dependency PNG
    png_path = out_dir / "dependency.png"
    make_static(G, stem, png_path)
    print(f"  ✓ Dependency diagram → {png_path}")


if __name__ == "__main__":
    graph_files = glob.glob(
        str(Path(__file__).resolve().parent.parent / "data" / "graphs" / "*.json")
    )
    if not graph_files:
        print("No graph JSON files found in data/graphs/")
    for gf in sorted(graph_files):
        process_graph(gf)
    print("\nDone. Outputs are in outputs/<video>/")
