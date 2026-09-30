"""
infer.py — Generate a floor plan from a JSON description.

Usage
-----
    # From an existing JSON file in training-data/
    python infer.py --json ../training-data/42.json

    # Generate 3 different layouts for the same description
    python infer.py --json ../training-data/42.json --num-samples 3

    # From a hand-written JSON (only needs "rooms" with types, areas, connections)
    python infer.py --json my_house.json

    # Save without displaying
    python infer.py --json ../training-data/42.json --output my_plan.png
"""

import os
import json
import argparse

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config as cfg
from model import FloorPlanCVAE
from torch_geometric.data import Data


# ── room colours ────────────────────────────────────────────────────────

ROOM_COLORS = {
    "living":     "#d9d9d9",
    "bedroom":    "#66c2a5",
    "bathroom":   "#fc8d62",
    "kitchen":    "#8da0cb",
    "balcony":    "#b3b3b3",
    "front_door": "#a63603",
    "storage":    "#FF8C69",
    "stair":      "#9e9ac8",
}


# ── convert JSON → PyG Data ────────────────────────────────────────────

def json_to_data(desc: dict, device: torch.device) -> Data:
    """Turn a simplified JSON description into a PyG Data object.

    The JSON only needs a ``rooms`` list.  Each room needs:
        id, type, area, connections[{target_id, type}]

    No ground-truth bounding boxes are required.
    """
    rooms = desc["rooms"]

    # map room string IDs → integer indices
    id_to_idx = {r["id"]: i for i, r in enumerate(rooms)}

    # ── node features ───────────────────────────────────────────────────
    max_area = max(r["area"] for r in rooms) or 1.0
    node_features = []

    for r in rooms:
        rtype = r["type"]
        type_idx = (cfg.ROOM_TYPES.index(rtype)
                    if rtype in cfg.ROOM_TYPES else 0)
        onehot = [0.0] * cfg.NUM_ROOM_TYPES
        onehot[type_idx] = 1.0
        area_norm = r["area"] / max_area
        node_features.append(onehot + [area_norm])

    # ── edges ───────────────────────────────────────────────────────────
    src, dst, edge_attrs = [], [], []
    seen = set()

    for r in rooms:
        u = id_to_idx[r["id"]]
        for conn in r.get("connections", []):
            tid = conn["target_id"]
            if tid not in id_to_idx:
                continue
            v = id_to_idx[tid]

            pair = (min(u, v), max(u, v))
            if pair in seen:
                continue
            seen.add(pair)

            etype = conn.get("type", "adjacency")
            if etype in cfg.EDGE_TYPES:
                etype_idx = cfg.EDGE_TYPES.index(etype)
            else:
                etype_idx = cfg.EDGE_TYPES.index("adjacency")

            onehot_e = [0.0] * cfg.NUM_EDGE_TYPES
            onehot_e[etype_idx] = 1.0

            # both directions (undirected)
            src.extend([u, v])
            dst.extend([v, u])
            edge_attrs.extend([onehot_e, onehot_e])

    if not edge_attrs:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, cfg.NUM_EDGE_TYPES), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data.to(device)


# ── drawing ─────────────────────────────────────────────────────────────

def draw_layout(bboxes, room_types, title="", ax=None):
    """Draw predicted bounding boxes as coloured rectangles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for (cx, cy, w, h), rtype in zip(bboxes, room_types):
        color = ROOM_COLORS.get(rtype, "#cccccc")
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.005",
            linewidth=2, edgecolor="black",
            facecolor=color, alpha=0.75,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, rtype.replace("_", "\n"),
                ha="center", va="center", fontsize=7, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a floor plan from a JSON description")
    parser.add_argument("--json", required=True,
                        help="Path to a JSON file describing the house")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of different layouts to generate "
                             "(each uses a different random seed)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save image to this path instead of displaying")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint "
                             "(default: checkpoints/best_model.pt)")
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    # ── load model ──────────────────────────────────────────────────────
    model_path = args.model or os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}")
        print("Run  python train.py  first.")
        return

    model = FloorPlanCVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # ── load JSON description ───────────────────────────────────────────
    with open(args.json) as f:
        desc = json.load(f)

    room_types = [r["type"] for r in desc["rooms"]]
    plan_id = desc.get("id", "?")
    print(f"Plan {plan_id}: {len(desc['rooms'])} rooms — "
          f"{', '.join(room_types)}")

    # ── convert to tensors ──────────────────────────────────────────────
    data = json_to_data(desc, device)

    # ── generate layouts ────────────────────────────────────────────────
    n = args.num_samples
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        with torch.no_grad():
            pred = model.generate(data)

        bboxes = pred.cpu().numpy()
        draw_layout(bboxes, room_types,
                    title=f"Generated layout (sample {i + 1})", ax=ax)

        # print coordinates
        print(f"\n── Sample {i + 1} ──")
        for room, (cx, cy, w, h) in zip(desc["rooms"], bboxes):
            print(f"  {room['id']:20s}  cx={cx:.3f}  cy={cy:.3f}  "
                  f"w={w:.3f}  h={h:.3f}")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {args.output}")
    else:
        plt.savefig(os.path.join(cfg.CHECKPOINT_DIR, "inference_output.png"),
                    dpi=150, bbox_inches="tight")
        print(f"\nSaved → {os.path.join(cfg.CHECKPOINT_DIR, 'inference_output.png')}")
        plt.show()


if __name__ == "__main__":
    main()
