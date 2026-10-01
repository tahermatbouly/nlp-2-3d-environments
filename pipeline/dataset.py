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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pickle
import json
from typing import List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch_geometric.data import Dataset, Data

from pipeline import config as cfg
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
    """PyTorch Geometric dataset that wraps one split of ResPlan.
    Uses lazy loading with caching to avoid upfront processing of all plans.
    """

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
                all_plans = pickle.load(f)
        else:
            all_plans = plans

        with open(cfg.SPLIT_JSON) as f:
            splits = json.load(f)

        split_ids = set(splits[split])
        # Filter plans by split
        split_plans = [p for p in all_plans if p["id"] in split_ids]

        # Preprocess each plan to extract lightweight info and validate
        self.plans_data: List[Tuple] = []  # each entry: (plan_dict, plan_bounds, max_area, node_features, edge_index, edge_attr)
        skipped = 0
        for plan in split_plans:
            plan = normalize_keys(plan)
            graph = plan.get("graph")
            if graph is None or len(graph.nodes) == 0:
                skipped += 1
                continue

            # Build node mapping
            nodes = list(graph.nodes(data=True))
            node_to_idx = {name: i for i, (name, _) in enumerate(nodes)}

            # ── collect bounds of every room for normalisation ──────────────
            all_bounds = []
            for _, d in nodes:
                geom = d["geometry"]
                if hasattr(geom, "bounds") and not geom.is_empty:
                    all_bounds.append(geom.bounds)   # (minx, miny, maxx, maxy)

            if not all_bounds:
                skipped += 1
                continue

            bounds_arr = np.array(all_bounds)
            plan_minx = bounds_arr[:, 0].min()
            plan_miny = bounds_arr[:, 1].min()
            plan_maxx = bounds_arr[:, 2].max()
            plan_maxy = bounds_arr[:, 3].max()
            plan_scale = max(plan_maxx - plan_minx, plan_maxy - plan_miny, 1.0)
            plan_bounds = (plan_minx, plan_miny, plan_scale)

            # ── node features and edges ───────────────────────────────────
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

            # Store precomputed data along with the plan dict for rasterization later
            self.plans_data.append((
                plan,                     # we need the plan dict for rasterization
                plan_bounds,
                max_area,
                torch.tensor(node_features, dtype=torch.float),
                edge_index,
                edge_attr
            ))

        if skipped:
            print(f"[dataset] Skipped {skipped} plans with no valid geometry.")

        # Cache for loaded data objects (to avoid reprocessing the same index multiple times)
        # Using LRU cache with maximum size to prevent memory issues
        self._cache_max_size = 150  # Increased from 50 to 150 for better hit rate
        self._cache = OrderedDict()

    # ── PyG interface ───────────────────────────────────────────────────
    def len(self) -> int:
        return len(self.plans_data)

    def get(self, idx: int) -> Data:
        # Check cache first
        if idx in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(idx)
            return self._cache[idx]

        # Retrieve precomputed data
        plan_dict, plan_bounds, max_area, node_features, edge_index, edge_attr = self.plans_data[idx]

        # Rasterize the plan to get target_image
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
            geom = plan_dict.get(category)
            if geom is None:
                continue
            # Convert geometry to mask (binary)
            mask = geometry_to_mask(geom, shape=(img_size, img_size), point_radius=2, line_thickness=2)
            # Get color for this category
            color_hex = CATEGORY_COLORS.get(category, "#ffffff")
            # Convert hex to RGB float in [0,1]
            color_hex = color_hex.lstrip('#')
            color_rgb = np.array([int(color_hex[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
            # Create color array for vectorized application
            color_array = np.ones((img_size, img_size, 3), dtype=np.float32) * np.array(color_rgb)
            # Apply color where mask is 255 (foreground) - vectorized across all channels
            canvas = np.where(mask[..., None] == 255, color_array, canvas)

        # Convert canvas to torch tensor [3, H, W] and normalize to [-1, 1] for diffusion
        target_image = torch.from_numpy(canvas).permute(2, 0, 1)  # [3, H, W]
        target_image = target_image * 2.0 - 1.0  # [0,1] -> [-1,1]
        # Add batch dimension for proper batching: [1, 3, H, W]
        target_image = target_image.unsqueeze(0)

        # ── assemble PyG Data object ─────────────────────────────────────
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            target_img=target_image,
            plan_id=plan_dict.get("id", -1),
            plan_bounds=torch.tensor(plan_bounds, dtype=torch.float),
        )

        # Cache the data object to avoid reprocessing
        self._cache[idx] = data
        # Remove oldest item if cache exceeds maximum size
        if len(self._cache) > self._cache_max_size:
            self._cache.popitem(last=False)

        return data