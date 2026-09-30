"""
dataset.py — Converts ResPlan floor plans into PyTorch Geometric Data objects.

For each floor plan it builds:
    x             (N, 9)   node features  = one-hot room type (8) + normalised area (1)
    edge_index    (2, 2E)  edges in both directions (undirected graph)
    edge_attr     (2E, 4)  one-hot edge type per directed edge
    y             (N, 4)   ground-truth bounding box per room (cx, cy, w, h) in [0, 1]
    plan_id       int      original plan ID
    plan_bounds   (3,)     (min_x, min_y, scale) for denormalising back to pixel coords
"""

import pickle
import json
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Dataset, Data

import config as cfg


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

        # ── node features & target bboxes ───────────────────────────────
        max_area = max(d["area"] for _, d in nodes) or 1.0

        node_features = []
        target_bboxes = []

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

            # ground-truth bounding box, normalised to [0, 1]
            geom = d["geometry"]
            if hasattr(geom, "bounds") and not geom.is_empty:
                minx, miny, maxx, maxy = geom.bounds
                cx = ((minx + maxx) / 2 - plan_minx) / plan_scale
                cy = ((miny + maxy) / 2 - plan_miny) / plan_scale
                w  = (maxx - minx) / plan_scale
                h  = (maxy - miny) / plan_scale
                target_bboxes.append([cx, cy, w, h])
            else:
                target_bboxes.append([0.5, 0.5, 0.05, 0.05])

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

        # ── assemble PyG Data object ────────────────────────────────────
        if not edge_attrs:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, cfg.NUM_EDGE_TYPES), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr  = torch.tensor(edge_attrs, dtype=torch.float)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(target_bboxes, dtype=torch.float),
            plan_id=plan["id"],
            plan_bounds=torch.tensor(
                [plan_minx, plan_miny, plan_scale], dtype=torch.float
            ),
        )

        return data
