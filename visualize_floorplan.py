#!/usr/bin/env python3
"""
visualize_floorplan.py — Standalone visualizer for ResPlan dataset floorplans.

This script does NOT require geopandas or cv2. It only relies on:
    matplotlib, shapely, networkx, numpy (standard in your virtual environment).

Features:
  - Visualize ground-truth geometric layout (walls, rooms, doors, windows, balconies, etc.)
  - Visualize room connectivity graph (nodes placed at room centroids, sized by area)
  - Visualize normalized bounding boxes (the format used by the model/pipeline)
  - Lookup plans by ID or index from ResPlan.pkl or training-data/
  - Save to image file (--output) or show interactively

Usage:
  # View plan ID 0 with layout and graph side-by-side:
  python visualize_floorplan.py --id 0

  # View plan index 5 in the dataset:
  python visualize_floorplan.py --index 5

  # Save to file without displaying GUI window:
  python visualize_floorplan.py --id 42 --output plan_42.png

  # Show only the architectural layout:
  python visualize_floorplan.py --id 0 --mode layout

  # Show all 3 views (Architecture, Room Graph, and Model Bounding Boxes):
  python visualize_floorplan.py --id 0 --mode all
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, GeometryCollection
)
from shapely.plotting import plot_polygon, plot_line


# ── Color Palette ─────────────────────────────────────────────────────────────

CATEGORY_COLORS: Dict[str, str] = {
    "wall":       "#2b2b2b",  # dark charcoal / black walls
    "living":     "#d9d9d9",  # light gray
    "bedroom":    "#66c2a5",  # teal green
    "bathroom":   "#fc8d62",  # warm orange
    "kitchen":    "#8da0cb",  # soft slate blue
    "balcony":    "#b3b3b3",  # mid gray
    "storage":    "#e5c494",  # light tan
    "stair":      "#9e9ac8",  # lavender
    "front_door": "#e41a1c",  # bright red
    "door":       "#e78ac3",  # pink
    "window":     "#a6d854",  # lime green
    "garden":     "#b8e186",  # light green
    "parking":    "#cbd5e8",  # cool gray
    "pool":       "#9ecae1",  # water blue
}

EDGE_COLORS: Dict[str, str] = {
    "via_door":    "#d95f02",
    "adjacency":   "#7570b3",
    "via_window":  "#1b9e77",
    "direct":      "#e7298a",
    "via_opening": "#e6ab02",
    "fallback":    "#a6761d",
}


# ── Geometry Extraction ───────────────────────────────────────────────────────

def get_geometries(geom_data: Any) -> List[Any]:
    """Safely extract individual geometric primitives."""
    if geom_data is None:
        return []
    if isinstance(geom_data, (Polygon, LineString, Point)):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    if isinstance(geom_data, (list, tuple)):
        res = []
        for g in geom_data:
            res.extend(get_geometries(g))
        return res
    return []


# ── Plotting Helpers ──────────────────────────────────────────────────────────

def plot_geometry_item(geom: Any, ax: plt.Axes, category: str):
    """Render a shapely geometry onto a matplotlib axis with proper styling."""
    color = CATEGORY_COLORS.get(category, "#cccccc")
    parts = get_geometries(geom)

    for part in parts:
        if isinstance(part, Polygon):
            if category == "wall":
                plot_polygon(
                    part, ax=ax, add_points=False,
                    facecolor=color, edgecolor=color,
                    alpha=0.95, linewidth=0.5
                )
            elif category in ("door", "front_door"):
                plot_polygon(
                    part, ax=ax, add_points=False,
                    facecolor=color, edgecolor="black",
                    alpha=0.9, linewidth=1.0
                )
            elif category == "window":
                plot_polygon(
                    part, ax=ax, add_points=False,
                    facecolor=color, edgecolor="#4d9221",
                    alpha=0.85, linewidth=0.8
                )
            else:
                plot_polygon(
                    part, ax=ax, add_points=False,
                    facecolor=color, edgecolor="#404040",
                    alpha=0.75, linewidth=0.75
                )
        elif isinstance(part, (LineString, MultiLineString)):
            lw = 3.0 if category in ("door", "front_door", "window") else 1.5
            plot_line(part, ax=ax, add_points=False, color=color, linewidth=lw)
        elif isinstance(part, Point):
            ax.plot(part.x, part.y, "o", color=color, markersize=5)


def render_architectural_plan(plan: Dict[str, Any], ax: plt.Axes, title: Optional[str] = None):
    """Draw full architectural floorplan (rooms, doors, windows, walls)."""
    # Draw order: backgrounds/rooms first, then openings/doors/windows, then walls on top
    draw_order = [
        "garden", "parking", "pool", "land",
        "living", "bedroom", "kitchen", "bathroom", "balcony", "storage", "stair",
        "door", "window", "front_door",
        "wall"
    ]

    present_categories = []
    for cat in draw_order:
        geom = plan.get(cat)
        if geom is not None:
            geoms = get_geometries(geom)
            if geoms:
                plot_geometry_item(geom, ax, cat)
                if cat not in present_categories:
                    present_categories.append(cat)

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Legend
    legend_patches = [
        mpatches.Patch(
            facecolor=CATEGORY_COLORS.get(k, "#cccccc"),
            edgecolor="none" if k == "wall" else "#333333",
            label=k.replace("_", " ").title()
        )
        for k in present_categories
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=8
        )


def render_graph_view(plan: Dict[str, Any], ax: plt.Axes, title: Optional[str] = None):
    """Draw the room connectivity graph overlaid on an understated floorplan outline."""
    # Underlay rooms with light transparency
    room_categories = ["living", "bedroom", "kitchen", "bathroom", "balcony", "storage", "stair"]
    for cat in room_categories:
        geom = plan.get(cat)
        if geom is not None:
            for part in get_geometries(geom):
                if isinstance(part, Polygon):
                    plot_polygon(
                        part, ax=ax, add_points=False,
                        facecolor="#f0f0f0", edgecolor="#b0b0b0",
                        alpha=0.6, linewidth=0.5
                    )

    # Walls underlay
    wall_geom = plan.get("wall")
    if wall_geom is not None:
        for part in get_geometries(wall_geom):
            if isinstance(part, Polygon):
                plot_polygon(
                    part, ax=ax, add_points=False,
                    facecolor="#333333", edgecolor="#333333",
                    alpha=0.3, linewidth=0.3
                )

    G = plan.get("graph")
    if G is None or len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No connectivity graph found", ha="center", va="center", transform=ax.transAxes)
        return

    # Compute node positions from room centroids
    pos = {}
    for n, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            c = geom.centroid
            pos[n] = (c.x, c.y)
        else:
            pos[n] = (0.0, 0.0)

    # Draw edges grouped by connection type
    edge_types = {}
    for u, v, d in G.edges(data=True):
        etype = d.get("type", "adjacency")
        edge_types.setdefault(etype, []).append((u, v))

    for etype, elist in edge_types.items():
        ecolor = EDGE_COLORS.get(etype, "#888888")
        style = "solid" if etype in ("via_door", "direct") else "dashed"
        nx.draw_networkx_edges(
            G, pos, edgelist=elist, ax=ax,
            edge_color=ecolor, width=2.0, style=style, alpha=0.85
        )

    # Draw nodes
    node_colors = []
    node_sizes = []
    labels = {}
    for n, d in G.nodes(data=True):
        rtype = d.get("type", "living")
        node_colors.append(CATEGORY_COLORS.get(rtype, "#aaaaaa"))
        area = float(d.get("area", 1000.0))
        # Scale node size reasonably
        node_sizes.append(max(200, min(1400, int(np.sqrt(max(0, area)) * 8))))
        # Clean label (e.g. bed_0)
        labels[n] = n.replace("bedroom", "bed").replace("bathroom", "bath").replace("front_door", "FD")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=1.5,
        alpha=0.95
    )
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight="bold", ax=ax)

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Edge legend
    edge_handles = [
        mpatches.Patch(color=EDGE_COLORS.get(k, "#888888"), label=k.replace("_", " ").title())
        for k in edge_types.keys()
    ]
    if edge_handles:
        ax.legend(
            handles=edge_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            title="Connections",
            fontsize=8
        )


def render_bounding_boxes(plan: Dict[str, Any], ax: plt.Axes, title: Optional[str] = None):
    """Draw room bounding boxes normalized into [0, 1] space as used by the model."""
    G = plan.get("graph")
    if G is None or len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No room nodes found", ha="center", va="center", transform=ax.transAxes)
        return

    nodes = list(G.nodes(data=True))
    all_bounds = []
    for _, d in nodes:
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "bounds") and not geom.is_empty:
            all_bounds.append(geom.bounds)

    if not all_bounds:
        ax.text(0.5, 0.5, "No room geometries available", ha="center", va="center", transform=ax.transAxes)
        return

    bounds_arr = np.array(all_bounds)
    minx, miny = bounds_arr[:, 0].min(), bounds_arr[:, 1].min()
    maxx, maxy = bounds_arr[:, 2].max(), bounds_arr[:, 3].max()
    scale = max(maxx - minx, maxy - miny, 1.0)

    for nid, d in nodes:
        rtype = d.get("type", "living")
        color = CATEGORY_COLORS.get(rtype, "#cccccc")
        geom = d.get("geometry")
        if geom is None or geom.is_empty:
            continue
        gx1, gy1, gx2, gy2 = geom.bounds
        cx = ((gx1 + gx2) / 2 - minx) / scale
        cy = ((gy1 + gy2) / 2 - miny) / scale
        w = (gx2 - gx1) / scale
        h = (gy2 - gy1) / scale

        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.004",
            linewidth=1.8, edgecolor="black",
            facecolor=color, alpha=0.8
        )
        ax.add_patch(rect)
        label_text = nid.replace("_", "\n")
        ax.text(cx, cy, label_text, ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (norm)")
    ax.set_ylabel("y (norm)")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)


# ── Loading Data ──────────────────────────────────────────────────────────────

def load_plan(data_pkl_path: str, plan_id: Optional[int] = None, plan_index: Optional[int] = None) -> Dict[str, Any]:
    """Load a plan dictionary from ResPlan.pkl by ID or dataset index."""
    if not os.path.exists(data_pkl_path):
        raise FileNotFoundError(f"Could not find dataset file: {data_pkl_path}")

    print(f"Loading dataset from {data_pkl_path}...")
    with open(data_pkl_path, "rb") as f:
        dataset = pickle.load(f)

    if plan_index is not None:
        if 0 <= plan_index < len(dataset):
            return dataset[plan_index]
        raise IndexError(f"Index {plan_index} out of range [0, {len(dataset)-1}]")

    if plan_id is not None:
        for plan in dataset:
            if plan.get("id") == plan_id:
                return plan
        raise ValueError(f"Plan ID {plan_id} not found in dataset.")

    # Default to first plan
    return dataset[0]


# ── Main CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Properly visualize a single floorplan from ResPlan.")
    parser.add_argument("--id", type=int, default=None, help="Plan ID to visualize (e.g. --id 0)")
    parser.add_argument("--index", type=int, default=None, help="Plan index in dataset (e.g. --index 10)")
    parser.add_argument("--dataset", type=str, default="ResPlan.pkl", help="Path to ResPlan.pkl (default: ResPlan.pkl)")
    parser.add_argument(
        "--mode", choices=["layout", "graph", "bbox", "side-by-side", "all"],
        default="side-by-side",
        help="Visualization mode: 'layout', 'graph', 'bbox', 'side-by-side' (default), or 'all'"
    )
    parser.add_argument("--output", type=str, default=None, help="Output image file path (e.g. plan_0.png)")
    parser.add_argument("--dpi", type=int, default=160, help="DPI for saved image (default: 160)")

    args = parser.parse_args()

    # Fallback to id 0 if neither id nor index is specified
    if args.id is None and args.index is None:
        args.id = 0

    plan = load_plan(args.dataset, plan_id=args.id, plan_index=args.index)
    pid = plan.get("id", "?")
    net_area = plan.get("net_area", 0.0)
    gross_area = plan.get("area", 0.0)

    info_str = f"Plan ID: {pid}  |  Net Area: {net_area:.1f} m²  |  Gross Area: {gross_area:.1f} m²"
    print(f"\nVisualizing {info_str}")

    # Set up figure according to mode
    if args.mode == "layout":
        fig, ax = plt.subplots(figsize=(8, 8))
        render_architectural_plan(plan, ax, title=f"Architectural Floorplan\n({info_str})")

    elif args.mode == "graph":
        fig, ax = plt.subplots(figsize=(8, 8))
        render_graph_view(plan, ax, title=f"Room Connectivity Graph\n({info_str})")

    elif args.mode == "bbox":
        fig, ax = plt.subplots(figsize=(7, 7))
        render_bounding_boxes(plan, ax, title=f"Model Normalized Bounding Boxes\n({info_str})")

    elif args.mode == "side-by-side":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        render_architectural_plan(plan, ax1, title="1. Architectural Layout")
        render_graph_view(plan, ax2, title="2. Room Connectivity Graph")
        fig.suptitle(info_str, fontsize=13, fontweight="bold", y=0.98)

    elif args.mode == "all":
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
        render_architectural_plan(plan, ax1, title="1. Architectural Layout")
        render_graph_view(plan, ax2, title="2. Room Connectivity Graph")
        render_bounding_boxes(plan, ax3, title="3. Model Bounding Boxes")
        fig.suptitle(info_str, fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout()

    if args.output:
        out_path = args.output
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Figure successfully saved to: {out_path}")
    else:
        default_out = f"plan_{pid}_visualization.png"
        plt.savefig(default_out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to: {default_out}")
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
