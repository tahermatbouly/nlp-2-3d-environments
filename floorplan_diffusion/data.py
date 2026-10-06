"""Dataset construction, normalisation, near-duplicate handling and batching.

Everything coordinate-related lives on the 256-unit canvas; the network sees
``(x - plan_centroid) / S`` with one dataset-wide S computed on the train split.
"""
from __future__ import annotations
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from torch.utils.data import Dataset

from .config import (EDGE_TYPE_TO_ID, MAX_ROOMS, MAX_VERTICES, MIN_ROOM_AREA,
                     NUM_ROOM_TYPES, ROOM_TYPE_TO_ID)
from .geometry import canonical_start, preprocess_polygon

# ----------------------------------------------------------------------------
# graph -> tensors (no coordinates; this is all the graph encoder may ever see)
# ----------------------------------------------------------------------------

def _node_type(d: Dict[str, Any]) -> str:
    return d.get("room_type", d.get("type"))


def _edge_type(d: Dict[str, Any]) -> str:
    return d.get("connection_type", d.get("type"))


def graph_to_arrays(G: nx.Graph) -> Optional[Dict[str, Any]]:
    """Room types, areas and typed edges of a floorplan graph."""
    names = list(G.nodes)
    if not 1 <= len(names) <= MAX_ROOMS:
        return None
    idx = {n: i for i, n in enumerate(names)}
    try:
        rtype = np.array([ROOM_TYPE_TO_ID[_node_type(G.nodes[n])] for n in names], dtype=np.int64)
        area = np.array([float(G.nodes[n]["area"]) for n in names], dtype=np.float32)
        edges, etypes = [], []
        for u, v, d in G.edges(data=True):
            edges.append((idx[u], idx[v]))
            etypes.append(EDGE_TYPE_TO_ID[_edge_type(d)])
    except KeyError:
        return None
    ei = np.array(edges, dtype=np.int64).reshape(-1, 2).T
    return dict(names=names, room_type=rtype, area=area, edge_index=ei,
                edge_type=np.array(etypes, dtype=np.int64))


def build_sample(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Plan dict -> numpy sample in *centred canvas units* (not yet scaled by S)."""
    G = plan["graph"]
    base = graph_to_arrays(G)
    if base is None:
        return None
    rings = []
    for n in base["names"]:
        ring = preprocess_polygon(G.nodes[n]["geometry"])
        if ring is None:
            return None
        rings.append(ring)
    from shapely.geometry import Polygon
    polys = [Polygon(r) for r in rings]
    if min(p.area for p in polys) < MIN_ROOM_AREA:
        return None
    union = unary_union(polys)
    c = np.array([union.centroid.x, union.centroid.y], dtype=np.float64)
    n = len(rings)
    coords = np.zeros((n, MAX_VERTICES, 2), dtype=np.float32)
    vmask = np.zeros((n, MAX_VERTICES), dtype=bool)
    for i, r in enumerate(rings):
        coords[i, :len(r)] = (r - c)
        vmask[i, :len(r)] = True
    # keep the graph's `area` attribute but re-measure from the cleaned polygon
    base["area"] = np.array([p.area for p in polys], dtype=np.float32)
    base.update(id=int(plan["id"]), coords=coords, vmask=vmask,
                nverts=vmask.sum(1).astype(np.int64), centroid=c.astype(np.float32))
    return base


# ----------------------------------------------------------------------------
# near-duplicate detection (rotation / flip / scale invariant)  — §26
# ----------------------------------------------------------------------------

def duplicate_descriptor(s: Dict[str, Any], per_type: int = 8) -> np.ndarray:
    """Sorted relative room areas per type + typed edge counts."""
    tot = float(s["area"].sum())
    parts = []
    for t in range(NUM_ROOM_TYPES):
        a = np.sort(s["area"][s["room_type"] == t] / tot)[::-1][:per_type]
        parts.append(np.pad(a, (0, per_type - len(a))))
    ec = np.bincount(s["edge_type"], minlength=len(EDGE_TYPE_TO_ID) + 1)[1:]
    return np.concatenate(parts + [ec.astype(np.float64) * 10.0])


def flag_duplicates(query: List[Dict], reference: List[Dict], radius: float = 0.004) -> List[bool]:
    """True where a query plan has a near-duplicate in ``reference`` (L-inf < radius)."""
    if not reference or not query:
        return [False] * len(query)
    tree = cKDTree(np.stack([duplicate_descriptor(s) for s in reference]))
    q = np.stack([duplicate_descriptor(s) for s in query])
    d, _ = tree.query(q, p=np.inf, distance_upper_bound=radius)
    return list(np.isfinite(d))


# ----------------------------------------------------------------------------
# cache building
# ----------------------------------------------------------------------------

def build_cache(pkl_path: str, split_path: str, cache_path: str, verbose: bool = True) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    with open(split_path) as f:
        splits = json.load(f)
    by_id = {int(p["id"]): p for p in data}
    aug = set(splits["augmented"])
    stats = {"dropped_invalid": 0, "dropped_aug": 0, "dropped_dup": {"val": 0, "test": 0}}
    out: Dict[str, List[Dict]] = {}
    for name in ("train", "val", "test"):
        samples = []
        for pid in splits[name]:
            if pid in aug:                      # augmented plans excluded everywhere
                stats["dropped_aug"] += 1
                continue
            if pid not in by_id:
                continue
            s = build_sample(by_id[pid])
            if s is None:
                stats["dropped_invalid"] += 1
                continue
            samples.append(s)
        out[name] = samples
        if verbose:
            print(f"[data] {name}: {len(samples)} plans")
    # remove near-duplicates from eval splits (Rules 26/27)
    flags = flag_duplicates(out["val"], out["train"])
    stats["dropped_dup"]["val"] = int(sum(flags))
    out["val"] = [s for s, f in zip(out["val"], flags) if not f]
    flags = flag_duplicates(out["test"], out["train"] + out["val"])
    stats["dropped_dup"]["test"] = int(sum(flags))
    out["test"] = [s for s, f in zip(out["test"], flags) if not f]
    # dataset-wide scale from TRAIN only: 99.9th percentile of |coordinate|, rounded up to 8
    allc = np.concatenate([s["coords"][s["vmask"]].reshape(-1) for s in out["train"]])
    S = float(np.ceil(np.percentile(np.abs(allc), 99.9) / 8.0) * 8.0)
    cache = dict(S=S, stats=stats, **out)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(cache, cache_path)
    if verbose:
        print(f"[data] S={S}  stats={stats}  sizes=" + ", ".join(f"{k}:{len(out[k])}" for k in out))
    return cache


def load_cache(pkl_path: str, split_path: str, cache_dir: str) -> Dict[str, Any]:
    path = os.path.join(cache_dir, "dataset.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    return build_cache(pkl_path, split_path, path)


# ----------------------------------------------------------------------------
# torch dataset / collate (normalisation by S happens here, in one place)
# ----------------------------------------------------------------------------

def area_features(area: torch.Tensor, S: float) -> torch.Tensor:
    """Node-area encoding fed to the graph encoder: [sqrt(A)/S, log(A)/10]."""
    a = area.clamp(min=1.0)
    return torch.stack([a.sqrt() / S, a.log() / 10.0], dim=-1)


def coords_to_rect(coords: torch.Tensor, vmask: torch.Tensor) -> torch.Tensor:
    """Axis-aligned bbox parameters (cx, cy, w, h) from polygon vertices."""
    big = 1e6
    m = vmask.unsqueeze(-1)
    lo = torch.where(m, coords, torch.full_like(coords, big)).amin(dim=-2)
    hi = torch.where(m, coords, torch.full_like(coords, -big)).amax(dim=-2)
    return torch.cat([(lo + hi) / 2, hi - lo], dim=-1)


class FloorplanDataset(Dataset):
    def __init__(self, samples: List[Dict], S: float, augment: bool = False):
        self.samples, self.S = samples, S
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        coords = s["coords"]
        vmask = s["vmask"]
        if self.augment:
            rot = np.random.randint(4)
            flip = np.random.rand() < 0.5
            c = coords.copy()
            for _ in range(rot):
                c = np.stack([-c[..., 1], c[..., 0]], axis=-1)
            if flip:
                c[..., 0] = -c[..., 0]
            for j in range(len(s["room_type"])):
                ring = c[j][vmask[j]]
                if flip:
                    ring = ring[::-1]
                c[j][:len(ring)] = canonical_start(ring)
            coords = c
            
        return dict(
            id=s["id"], n=len(s["room_type"]),
            room_type=torch.as_tensor(s["room_type"]),
            area=torch.as_tensor(s["area"]),
            edge_index=torch.as_tensor(s["edge_index"]),
            edge_type=torch.as_tensor(s["edge_type"]),
            coords=torch.as_tensor(coords) / self.S,
            vmask=torch.as_tensor(vmask),
            nverts=torch.as_tensor(s["nverts"]),
        )


def collate(items: List[Dict]) -> Dict[str, torch.Tensor]:
    B = len(items)
    N = max(it["n"] for it in items)
    V = MAX_VERTICES
    b = dict(
        room_type=torch.zeros(B, N, dtype=torch.long),
        room_mask=torch.zeros(B, N, dtype=torch.bool),
        area=torch.ones(B, N),
        adj=torch.zeros(B, N, N, dtype=torch.long),
        coords=torch.zeros(B, N, V, 2),
        vmask=torch.zeros(B, N, V, dtype=torch.bool),
        nverts=torch.zeros(B, N, dtype=torch.long),
        ids=torch.tensor([it["id"] for it in items]),
    )
    for k, it in enumerate(items):
        n = it["n"]
        b["room_type"][k, :n] = it["room_type"]
        b["room_mask"][k, :n] = True
        b["area"][k, :n] = it["area"]
        b["coords"][k, :n] = it["coords"]
        b["vmask"][k, :n] = it["vmask"]
        b["nverts"][k, :n] = it["nverts"]
        ei, et = it["edge_index"], it["edge_type"]
        if ei.numel():
            b["adj"][k, ei[0], ei[1]] = et
            b["adj"][k, ei[1], ei[0]] = et
    b["rect"] = coords_to_rect(b["coords"], b["vmask"])
    b["rect"] = b["rect"] * b["room_mask"].unsqueeze(-1)
    return b


def graph_batch(G_list: List[nx.Graph], S: float) -> Dict[str, torch.Tensor]:
    """Conditioning-only batch from NetworkX graphs (inference: no geometry)."""
    items = []
    for G in G_list:
        a = graph_to_arrays(G)
        if a is None:
            raise ValueError("graph has unsupported node/edge types or too many rooms")
        n = len(a["room_type"])
        items.append(dict(id=-1, n=n, room_type=torch.as_tensor(a["room_type"]),
                          area=torch.as_tensor(a["area"]),
                          edge_index=torch.as_tensor(a["edge_index"]),
                          edge_type=torch.as_tensor(a["edge_type"]),
                          coords=torch.zeros(n, MAX_VERTICES, 2),
                          vmask=torch.zeros(n, MAX_VERTICES, dtype=torch.bool),
                          nverts=torch.zeros(n, dtype=torch.long)))
    return collate(items)
