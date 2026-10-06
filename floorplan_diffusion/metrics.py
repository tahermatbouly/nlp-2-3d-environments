"""Evaluation as a *geometric generator* (plan §27): room-level + floorplan-level
metrics and the Constraint Satisfaction Rate.  Never coordinate MSE alone.
"""
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .generate import (Candidate, build_candidate, check_polygons, constraints_satisfied,
                       generate_floorplans, sample_to_graph)


def gt_polygons(s: Dict[str, Any]) -> List[Polygon]:
    """Ground-truth room polygons (centred canvas units) from a cached sample."""
    return [Polygon(s["coords"][i][s["vmask"][i]]) for i in range(len(s["room_type"]))]


def _centred(polys: List[Polygon]) -> List[Polygon]:
    good = [p for p in polys if p is not None and not p.is_empty and p.is_valid]
    if not good:
        return polys
    c = unary_union(good).centroid
    return [affinity.translate(p, -c.x, -c.y) if p is not None else None for p in polys]


def room_level(cand: Candidate, s: Dict[str, Any]) -> Dict[str, List[float]]:
    """Per-room comparison with the ground truth after aligning layout centroids."""
    gt = _centred(gt_polygons(s))
    pr = _centred(cand.polygons)
    out = {"area_err": [], "centroid_err": [], "iou": [], "coord_err": [], "vcount_acc": []}
    for i, (g, p) in enumerate(zip(gt, pr)):
        req = float(s["area"][i])
        if p is None or p.is_empty or not p.is_valid:
            out["area_err"].append(1.0); out["centroid_err"].append(np.nan)
            out["iou"].append(0.0); out["vcount_acc"].append(0.0)
            continue
        out["area_err"].append(abs(p.area - req) / max(req, 1.0))
        out["centroid_err"].append(float(np.hypot(p.centroid.x - g.centroid.x, p.centroid.y - g.centroid.y)))
        inter = p.intersection(g).area
        out["iou"].append(inter / max(p.union(g).area, 1e-6))
        n_p, n_g = len(p.exterior.coords) - 1, len(g.exterior.coords) - 1
        out["vcount_acc"].append(float(n_p == n_g))
        if n_p == n_g:
            out["coord_err"].append(float(np.linalg.norm(np.asarray(p.exterior.coords)[:-1]
                                                         - np.asarray(g.exterior.coords)[:-1], axis=1).mean()))
    return out


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows]
        if not vals or np.isnan(vals).all():
            out[k] = np.nan
        else:
            out[k] = float(np.nanmean(vals))
    return out


def evaluate_ground_truth(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run the *same* checker on the real plans: the achievable upper bound for CSR."""
    rows = []
    for s in samples:
        G = sample_to_graph(s)
        c = check_polygons(G, gt_polygons(s))
        rows.append(dict(csr=float(constraints_satisfied(c)), csr_strict=float(constraints_satisfied(c, True)),
                         edge_recall=c["edge_recall"], spurious_rate=c["spurious_rate"],
                         invalid_polygon=float(not c["polygons_ok"]), overlap_ok=float(c["overlap_ok"]),
                         area_ok=float(c["area_ok"])))
    return summarise(rows)


def evaluate_model(model, samples: List[Dict[str, Any]], K: int = 16, steps: int = 50, eta: float = 0.0,
                   device: str = "cpu", seed: int = 0, batch_size: int = 32, guidance: float = 1.0) -> Dict[str, Any]:
    graphs = [sample_to_graph(s) for s in samples]
    all_cands = generate_floorplans(model, graphs, K=K, steps=steps, eta=eta, device=device,
                                    seed=seed, batch_size=batch_size, guidance=guidance)
    floor, rooms, rooms_best = [], [], []
    for s, cands in zip(samples, all_cands):
        top = cands[0]
        ck = [c.checks for c in cands]
        floor.append(dict(
            csr_top1=float(constraints_satisfied(top.checks)),
            csr_top1_strict=float(constraints_satisfied(top.checks, True)),
            csr_any_of_k=float(any(constraints_satisfied(c) for c in ck)),
            csr_mean_over_k=float(np.mean([constraints_satisfied(c) for c in ck])),
            invalid_polygon_pct=100 * float(np.mean([not c["polygons_ok"] for c in ck])),
            room_invalid_pct=100 * float(np.mean([c["n_invalid"] / c["n_rooms"] for c in ck])),
            overlap_viol_pct=100 * float(np.mean([not c["overlap_ok"] for c in ck])),
            area_ok_pct=100 * float(np.mean([c["area_ok"] for c in ck])),
            connectivity_acc=float(np.mean([c["edge_recall"] for c in ck])),
            spurious_edge_rate=float(np.mean([c["spurious_rate"] for c in ck])),
            topology_consistent_pct=100 * float(np.mean([c["topology_ok"] and c["no_spurious"] for c in ck])),
            geometric_validity_pct=100 * float(np.mean([c["polygons_ok"] and c["overlap_ok"] for c in ck])),
        ))
        r = room_level(top, s)
        rooms.append({k: float(np.nanmean(v)) if len(v) and not np.isnan(v).all() else np.nan for k, v in r.items()})
        # best-of-K by mean IoU (reference similarity only; many valid layouts exist per graph)
        best = max((room_level(c, s) for c in cands), key=lambda d: np.mean(d["iou"]))
        rooms_best.append({k: float(np.nanmean(v)) if len(v) and not np.isnan(v).all() else np.nan for k, v in best.items()})
    return dict(floorplan=summarise(floor),
                room_top1=summarise(rooms),
                room_best_of_k=summarise(rooms_best),
                n_plans=len(samples), K=K, steps=steps)
