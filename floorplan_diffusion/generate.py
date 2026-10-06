"""Sampling, geometric validation, ranking and Shapely/NetworkX reconstruction
(plan §20-24).  This is the only place where network output turns into Shapely.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from .config import (AREA_REL_TOL, CANVAS, CONNECT_TOL, EDGE_TYPES, MAX_EXTENT_FACTOR,
                     OVERLAP_EXEMPT_TYPES, OVERLAP_REL_TOL, ROOM_TYPES, SCORE_WEIGHTS, SPURIOUS_TOL)
from .data import graph_batch, graph_to_arrays
from .geometry import clean_ring, signed_area


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def sample_to_graph(s: Dict[str, Any]) -> nx.Graph:
    """Conditioning graph (type, area, typed edges; *no* geometry) from a cached sample."""
    G = nx.Graph()
    names = [f"{ROOM_TYPES[t]}_{i}" for i, t in enumerate(s["room_type"])]
    for n, t, a in zip(names, s["room_type"], s["area"]):
        G.add_node(n, type=ROOM_TYPES[int(t)], area=float(a))
    for (u, v), e in zip(s["edge_index"].T, s["edge_type"]):
        G.add_edge(names[u], names[v], type=EDGE_TYPES[int(e) - 1])
    return G


def _to_polygon(ring: np.ndarray) -> Optional[Polygon]:
    if ring is None or len(ring) < 3:
        return None
    try:
        p = Polygon(ring)
    except Exception:
        return None
    return p


@dataclass
class Candidate:
    graph: nx.Graph                          # reconstructed floorplan graph (Shapely geometry on nodes)
    polygons: List[Optional[Polygon]]
    checks: Dict[str, Any] = field(default_factory=dict)
    score: float = float("inf")
    valid: bool = False                      # satisfies all required constraints (§22)


# ----------------------------------------------------------------------------
# validation (§22)
# ----------------------------------------------------------------------------

def check_polygons(G_in: nx.Graph, polys: Sequence[Optional[Polygon]]) -> Dict[str, Any]:
    names = list(G_in.nodes)
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    ok_poly = [p is not None and (not p.is_empty) and p.is_valid and p.area > 1.0 for p in polys]
    n_invalid = n - sum(ok_poly)
    req = np.array([G_in.nodes[m]["area"] for m in names], dtype=float)
    area = np.array([p.area if ok else 0.0 for p, ok in zip(polys, ok_poly)])
    area_rel = np.abs(area - req) / np.maximum(req, 1.0)
    area_ok = bool((area_rel <= AREA_REL_TOL).all())

    # overlap
    ov_pairs, worst = [], 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if not (ok_poly[i] and ok_poly[j]):
                continue
            if G_in.nodes[names[i]]["type"] in OVERLAP_EXEMPT_TYPES or G_in.nodes[names[j]]["type"] in OVERLAP_EXEMPT_TYPES:
                continue
            inter = polys[i].intersection(polys[j]).area
            r = inter / max(min(polys[i].area, polys[j].area), 1.0)
            ov_pairs.append(r)
            worst = max(worst, r)
    overlap_ok = worst <= OVERLAP_REL_TOL
    overlap_sum = float(np.sum(ov_pairs)) if ov_pairs else 0.0

    # connectivity / topology
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            if ok_poly[i] and ok_poly[j]:
                dist[i, j] = dist[j, i] = polys[i].distance(polys[j])
    req_edges, hit = 0, 0
    missing = []
    for u, v, d in G_in.edges(data=True):
        i, j = idx[u], idx[v]
        req_edges += 1
        if dist[i, j] <= CONNECT_TOL[d["type"]]:
            hit += 1
        else:
            missing.append((u, v))
    spurious, non_pairs = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if not G_in.has_edge(names[i], names[j]):
                non_pairs += 1
                if dist[i, j] <= SPURIOUS_TOL:
                    spurious += 1

    union = unary_union([p for p, ok in zip(polys, ok_poly) if ok]) if any(ok_poly) else None
    extent = 0.0
    compact = 1.0
    if union is not None and not union.is_empty:
        x0, y0, x1, y1 = union.bounds
        extent = max(x1 - x0, y1 - y0)
        hull = union.convex_hull.area
        compact = 1.0 - union.area / max(hull, 1e-6)
    dims_ok = extent <= MAX_EXTENT_FACTOR * CANVAS and extent > 0

    return dict(
        n_rooms=n, n_invalid=n_invalid, polygons_ok=n_invalid == 0,
        area_rel=area_rel, area_err=float(area_rel.mean()), area_ok=area_ok,
        worst_overlap=worst, overlap_sum=overlap_sum, overlap_ok=bool(overlap_ok),
        req_edges=req_edges, edges_hit=hit, edge_recall=hit / max(req_edges, 1),
        missing=missing, topology_ok=len(missing) == 0,
        spurious=spurious, spurious_rate=spurious / max(non_pairs, 1), no_spurious=spurious == 0,
        extent=extent, dims_ok=bool(dims_ok), compactness=float(compact), dist=dist,
    )


def constraints_satisfied(c: Dict[str, Any], strict: bool = False) -> bool:
    ok = c["polygons_ok"] and c["area_ok"] and c["overlap_ok"] and c["topology_ok"] and c["dims_ok"]
    return bool(ok and (c["no_spurious"] if strict else True))


def score_candidate(c: Dict[str, Any]) -> float:
    w = SCORE_WEIGHTS
    topo = (1 - c["edge_recall"]) + c["spurious_rate"]
    return float(w["area"] * c["area_err"] + w["overlap"] * c["overlap_sum"] + w["topology"] * topo
                 + w["invalid"] * c["n_invalid"] + w["compactness"] * c["compactness"])


# ----------------------------------------------------------------------------
# reconstruction (§24)
# ----------------------------------------------------------------------------

def build_candidate(G_in: nx.Graph, rings: List[np.ndarray]) -> Candidate:
    """Rings (canvas units, centred) -> validated Candidate with a NetworkX floorplan graph."""
    # invalid polygons are *reported* by the checks, never silently repaired
    polys = [_to_polygon(r) for r in rings]
    # shift into the positive quadrant (translation only)
    good = [p for p in polys if p is not None and not p.is_empty]
    if good:
        x0 = min(p.bounds[0] for p in good)
        y0 = min(p.bounds[1] for p in good)
        from shapely import affinity
        polys = [affinity.translate(p, 8 - x0, 8 - y0) if p is not None else None for p in polys]
    checks = check_polygons(G_in, polys)
    out = nx.Graph()
    for (n, d), p in zip(G_in.nodes(data=True), polys):
        out.add_node(n, type=d["type"], area=float(p.area) if p is not None else 0.0,
                     requested_area=d["area"],
                     geometry=p if p is not None else Polygon())
    for u, v, d in G_in.edges(data=True):
        out.add_edge(u, v, type=d["type"])
    cand = Candidate(graph=out, polygons=polys, checks=checks)
    cand.valid = constraints_satisfied(checks)
    cand.score = score_candidate(checks)
    return cand


def to_plan_dict(cand: Candidate, plan_id: int = -1) -> Dict[str, Any]:
    """ResPlan-style plan dict so ``resplan_utils.plot_plan_and_graph`` can draw it."""
    from shapely.ops import nearest_points
    from shapely.geometry import Point
    
    plan: Dict[str, Any] = {"id": plan_id, "graph": cand.graph}
    for t in ROOM_TYPES:
        geoms = [d["geometry"] for _, d in cand.graph.nodes(data=True)
                 if d["type"] == t and not d["geometry"].is_empty]
        if geoms:
            geoms = [g if g.is_valid else make_valid(g) for g in geoms]   # plotting only
            u = unary_union(geoms)
            plan[t] = u if isinstance(u, MultiPolygon) else MultiPolygon([u]) if isinstance(u, Polygon) else u
            
    # Procedural Doors
    doors = []
    for u, v, data in cand.graph.edges(data=True):
        if data.get("type") == "via_door":
            g_u = cand.graph.nodes[u].get("geometry")
            g_v = cand.graph.nodes[v].get("geometry")
            if g_u and g_v and not g_u.is_empty and not g_v.is_empty:
                try:
                    p_u, p_v = nearest_points(g_u, g_v)
                    midpt = Point((p_u.x + p_v.x) / 2, (p_u.y + p_v.y) / 2)
                    doors.append(midpt.buffer(2.0, cap_style=3))
                except Exception:
                    pass
    if doors:
        plan["door"] = unary_union(doors)
            
    # Procedural Walls and Inner boundary
    allg = [d["geometry"] for _, d in cand.graph.nodes(data=True) if not d["geometry"].is_empty]
    if allg:
        inner = unary_union([g if g.is_valid else make_valid(g) for g in allg])
        plan["inner"] = inner
        try:
            wall = inner.buffer(1.5, join_style=2).difference(inner.buffer(-0.5))
            if not wall.is_empty:
                plan["wall"] = wall
        except Exception:
            pass
            
    return plan


# ----------------------------------------------------------------------------
# sampling pipeline (§20-23)
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_floorplans(model, graphs: List[nx.Graph], K: int = 8, steps: int = 50, eta: float = 0.0,
                        device: str = "cpu", seed: int = 0, batch_size: int = 32,
                        sample_counts: bool = False, drop_invalid: bool = False,
                        snap_grid: float = 2.0) -> List[List[Candidate]]:
    """For every input graph draw K samples, validate and rank them (best first).

    ``drop_invalid=True`` rejects candidates that violate the constraints (§22);
    by default they are kept (flagged ``valid=False``) but ranked after valid ones.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    flat = [(gi, G) for gi, G in enumerate(graphs) for _ in range(K)]
    results: List[List[Candidate]] = [[] for _ in graphs]
    for s in range(0, len(flat), batch_size):
        chunk = flat[s:s + batch_size]
        batch = graph_batch([G for _, G in chunk], model.S)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.sample(batch, steps=steps, eta=eta, sample_counts=sample_counts, generator=gen)
        P, vm = model.to_polygons(out["x"], out["tok_mask"])
        P, vm = P.cpu().numpy().astype(np.float64), vm.cpu().numpy()
        for b, (gi, G) in enumerate(chunk):
            n = len(G)
            rings = []
            for i in range(n):
                r = P[b, i][vm[b, i]]
                if snap_grid > 0:
                    r = np.round(r / snap_grid) * snap_grid
                rings.append(clean_ring(r) if len(r) >= 3 else r)
            results[gi].append(build_candidate(G, rings))
    for gi in range(len(results)):
        cands = results[gi]
        if drop_invalid:
            cands = [c for c in cands if c.valid]
        results[gi] = sorted(cands, key=lambda c: (not c.valid, c.score))
    return results
