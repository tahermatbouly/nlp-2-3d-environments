"""
dataset.py — Converts ResPlan floor plans into PyTorch Geometric Data objects with rasterized floorplan targets.

For each floor plan it builds:
    x             (N, 9)   node features  = one-hot room type (8) + normalised area (1)
    edge_index    (2, 2E)  edges in both directions (undirected graph)
    edge_attr     (2E, 4)  one-hot edge type per directed edge
    target_image  (C, H, W) rasterized floorplan segmentation mask (C categories)
    plan_id       int      original plan ID
    plan_bounds   (3,)     (min_x, min_y, scale) for denormalising back to pixel coords
"""

import pickle
import json
from typing import List, Optional
import sys
import os

import numpy as np
import torch
from torch_geometric.data import Dataset, Data

import config as cfg

# Add the parent directory of this file (i.e., the root of the project) to sys.path
# so that we can import resplan_utils which is located in the root.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from resplan_utils import geometry_to_mask, CATEGORY_COLORS, normalize_keys


# Define segmentation categories (must match the keys we rasterize)
SEGMENTATION_CATEGORIES = [
    "wall",          # 0
    "living",        # 1
    "bedroom",       # 2
    "bathroom",      # 3
    "kitchen",       # 4
    "door",          # 5
    "window",        # 6
    "front_door",    # 7
    "balcony",       # 8
    "storage",       # 9
    "stair",         # 10
    # Optional: garden, parking, pool, land - we can add if needed
    # For now, we focus on the main architectural elements.
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(SEGMENTATION_CATEGORIES)}
NUM_SEGMENTATION_CLASSES = len(SEGMENTATION_CATEGORIES)


class FloorPlanDataset(Dataset):
    """PyTorch Geometric dataset that wraps one split of ResPlan."""

    def __init__(self, split: str = "train", plans: Optional[list] = None):
        """
        Parameters
        ----------
        split : str
            One of "train", "val", "test", "augmented".
        plans : list, optional
            Pre-loaded list of plan dicts (avoids reloading the .pkl).
            If None the .pkl is loaded from disk.
        """
        super().__init__()

        # ── load raw plans ──────────────────────────────────────────────
        if plans is None:
            with open(cfg.DATA_PKL, "rb") as f:
                plans = pickle.load(f)

        with open(cfg.SPLIT_JSON) as f:
            splits = json.load(f)

        split_ids = set(splits[split])
        self.plans = [p for p in plans if p["id"] in split_ids]

        # ── convert every plan to a PyG Data object ─────────────────────
        self.data_list: List[Data] = []
        skipped = 0
        for plan in self.plans:
            data = self._process_plan(plan)
            if data is not None:
                self.data_list.append(data)
            else:
                skipped += 1

        if skipped:
            print(f"[dataset] Skipped {skipped} plans with no valid geometry.")

    # ── PyG interface ───────────────────────────────────────────────────
    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    # ── Per-plan processing ─────────────────────────────────────────────
    def _process_plan(self, plan: dict) -> Optional[Data]:
        plan = normalize_keys(plan)
        graph = plan["graph"]
        nodes = list(graph.nodes(data=True))

        if len(nodes) == 0:
            return None

        # Build  node-name → integer index  mapping
        node_to_idx = {name: i for i, (name, _) in enumerate(nodes)}

        # ── collect bounds of every room for normalisation ──────────────
        all_bounds = []
        for _, d in nodes:
            geom = d["geometry"]
            if hasattr(geom, "bounds") and not geom.is_empty:
                all_bounds.append(geom.bounds)   # (minx, miny, maxx, maxy)

        if not all_bounds:
            return None

        bounds_arr = np.array(all_bounds)
        plan_minx = bounds_arr[:, 0].min()
        plan_miny = bounds_arr[:, 1].min()
        plan_maxx = bounds_arr[:, 2].max()
        plan_maxy = bounds_arr[:, 3].max()
        plan_scale = max(plan_maxx - plan_minx, plan_maxy - plan_miny, 1.0)

        # ── node features (same as before) ───────────────────────────────
        max_area = max(d["area"] for _, d in nodes) or 1.0

        node_features = []
        for _, d in nodes:
            # one-hot room type (8 dims)
            rtype = d["type"]
            type_idx = (cfg.ROOM_TYPES.index(rtype)
                        if rtype in cfg.ROOM_TYPES else 0)
            onehot = [0.0] * cfg.NUM_ROOM_TYPES
            onehot[type_idx] = 1.0

            # normalised area (1 dim)
            area_norm = d["area"] / max_area

            node_features.append(onehot + [area_norm])

        # ── edges (both directions for the undirected graph) ────────────
        src_list, dst_list, edge_attrs = [], [], []

        for u, v, d in graph.edges(data=True):
            if u not in node_to_idx or v not in node_to_idx:
                continue

            etype = d.get("type", "adjacency")
            if etype in cfg.EDGE_TYPES:
                etype_idx = cfg.EDGE_TYPES.index(etype)
            else:
                # unknown edge type (e.g. "fallback", "via_opening") → adjacency
                etype_idx = cfg.EDGE_TYPES.index("adjacency")

            onehot_e = [0.0] * cfg.NUM_EDGE_TYPES
            onehot_e[etype_idx] = 1.0

            ui, vi = node_to_idx[u], node_to_idx[v]
            src_list.extend([ui, vi])
            dst_list.extend([vi, ui])
            edge_attrs.extend([onehot_e, onehot_e])

        # ── assemble edge tensors ───────────────────────────────────────
        if not edge_attrs:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, cfg.NUM_EDGE_TYPES), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # ── rasterize the plan to a segmentation mask ───────────────────
        # We'll create a multi-channel mask where each channel is a binary mask for a category.
        # Alternatively, we could create a single-channel mask with category indices.
        # We'll use single-channel with category indices for simplicity (cross-entropy loss).
        # But note: diffusion models often predict noise in the same space as the input.
        # We'll treat the segmentation mask as a tensor of long integers (class indices).
        # However, adding noise to class indices doesn't make sense.
        # Therefore, we might need to predict the continuous mask (probabilities) or use a different approach.
        # Alternatively, we can generate an RGB image and predict noise in RGB space.
        # Let's change strategy: generate an RGB image where each category has a fixed color.
        # We can then treat the image as a continuous tensor [3, H, W] and predict noise.
        # We'll use the CATEGORY_COLORS to map each category to an RGB color.
        # We'll create an RGB image by filling each pixel with the color of the topmost category.
        # We'll define a drawing order (background to foreground) to handle overlaps.

        # Drawing order (background first): land, garden, parking, pool, wall, then rooms, then doors/windows, etc.
        # We'll follow the order from visualize_floorplan.py's render_architectural_plan.
        draw_order = [
            "garden", "parking", "pool", "land",
            "wall",
            "living", "bedroom", "kitchen", "bathroom", "balcony", "storage", "stair",
            "door", "window", "front_door"
        ]

        # Initialize RGB image with background color (e.g., white or light gray)
        img_size = cfg.IMAGE_SIZE if hasattr(cfg, 'IMAGE_SIZE') else 256
        canvas = np.ones((img_size, img_size, 3), dtype=np.float32) * 0.9  # light gray background

        # Rasterize each category onto the canvas
        for category in draw_order:
            geom = plan.get(category)
            if geom is None:
                continue
            # Convert geometry to mask (binary)
            mask = geometry_to_mask(geom, shape=(img_size, img_size), point_radius=2, line_thickness=2)
            # Get color for this category
            color_hex = CATEGORY_COLORS.get(category, "#ffffff")
            # Convert hex to RGB float in [0,1]
            color_hex = color_hex.lstrip('#')
            color_rgb = np.array([int(color_hex[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
            # Apply color where mask is 255 (foreground)
            for c in range(3):
                canvas[:, :, c] = np.where(mask == 255, color_rgb[c], canvas[:, :, c])

        # Convert canvas to torch tensor [3, H, W] and normalize to [-1, 1] for diffusion
        # Diffusion models often work with inputs in [-1, 1].
        target_image = torch.from_numpy(canvas).permute(2, 0, 1)  # [3, H, W]
        target_image = target_image * 2.0 - 1.0  # [0,1] -> [-1,1]

        # ── assemble PyG Data object ─────────────────────────────────────
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_image=target_image,
            plan_id=plan["id"],
            plan_bounds=torch.tensor(
                [plan_minx, plan_miny, plan_scale], dtype=torch.float
            ),
        )

        return data