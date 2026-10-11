"""Sampling, geometric validation, ranking and Shapely/NetworkX reconstruction
(plan §20-24).  This is the only place where network output turns into Shapely.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

from .config import (AREA_REL_TOL, CANVAS, CONNECT_TOL, EDGE_TYPES, MAX_EXTENT_FACTOR,
                     OVERLAP_EXEMPT_TYPES, OVERLAP_REL_TOL, ROOM_TYPES, SCORE_WEIGHTS, SPURIOUS_TOL)
from .data import graph_batch, graph_to_arrays
from .geometry import clean_ring, signed_area

MIN_ROOM_AREAS = {
    "living": 2500.0,
    "bedroom": 1500.0,
    "kitchen": 1000.0,
    "bathroom": 300.0,
    "balcony": 300.0,
    "storage": 200.0,
    "front_door": 40.0,
    "stair": 300.0,
}


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

def _extract_poly(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == 'Polygon':
        return geom
    if geom.geom_type == 'MultiPolygon':
        return max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type == 'GeometryCollection':
        polys_only = [g for g in geom.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if polys_only:
            best = max(polys_only, key=lambda g: g.area)
            return max(best.geoms, key=lambda g: g.area) if best.geom_type == 'MultiPolygon' else best
    return None


def simplify_orthogonal_living(poly, max_v=8):
    if poly is None or poly.is_empty:
        return poly
    if not poly.is_valid:
        poly = make_valid(poly)
    coords = list(poly.exterior.coords)[:-1]
    if 4 <= len(coords) <= max_v:
        return poly
    simp = poly
    for tol in [0.5, 1.0, 2.0, 4.0, 8.0]:
        s = poly.simplify(tol)
        if s is not None and not s.is_empty and s.is_valid and s.geom_type == 'Polygon':
            c = list(s.exterior.coords)[:-1]
            if 4 <= len(c) <= max_v:
                return s
            simp = s
    return simp


def force_orthogonal_polygon(poly):
    if poly is None or poly.is_empty:
        return poly
    coords = list(poly.exterior.coords)
    if len(coords) < 3:
        return poly
    coords = coords[:-1]
    new_coords = [list(coords[0])]
    for i in range(1, len(coords)):
        prev = new_coords[-1]
        curr = list(coords[i])
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        if dx < dy:
            curr[0] = prev[0]
        else:
            curr[1] = prev[1]
        if curr != prev:
            new_coords.append(curr)
    last = new_coords[-1]
    first = new_coords[0]
    if last[0] != first[0] and last[1] != first[1]:
        if abs(last[0] - first[0]) < abs(last[1] - first[1]):
            new_coords.append([first[0], last[1]])
        else:
            new_coords.append([last[0], first[1]])
    try:
        return Polygon(new_coords)
    except Exception:
        return poly


def relax_box_collisions(boxes, G_in=None, node_names=None, max_iters=80, padding=0.05):
    """Multi-body collision relaxation with topology spring attraction (MTV).
    Preserves exact box dimensions (w, h) and areas while driving overlaps to 0.0
    and pulling connected rooms together so their distance <= CONNECT_TOL."""
    boxes = np.array(boxes, dtype=np.float64)
    n = len(boxes)
    for it in range(max_iters):
        # 1. Spring pull along graph edges to satisfy connectivity (first half only)
        if G_in is not None and node_names is not None and it < (max_iters // 2):
            for u, v, d in G_in.edges(data=True):
                if u in node_names and v in node_names:
                    i, j = node_names.index(u), node_names.index(v)
                    cxi, cxj = (boxes[i, 0] + boxes[i, 2]) / 2.0, (boxes[j, 0] + boxes[j, 2]) / 2.0
                    cyi, cyj = (boxes[i, 1] + boxes[i, 3]) / 2.0, (boxes[j, 1] + boxes[j, 3]) / 2.0
                    p_i, p_j = box(*boxes[i]), box(*boxes[j])
                    d_val = p_i.distance(p_j)
                    tol = CONNECT_TOL.get(d["type"], 5.0)
                    if d_val > tol:
                        pull = min(d_val - tol + 0.5, 4.0) / 2.0
                        angle = np.arctan2(cyj - cyi, cxj - cxi)
                        dx = pull * np.cos(angle)
                        dy = pull * np.sin(angle)
                        boxes[i, [0, 2]] += dx; boxes[i, [1, 3]] += dy
                        boxes[j, [0, 2]] -= dx; boxes[j, [1, 3]] -= dy

        # 2. 2D MTV Collision resolution along center-displacement vector (all iterations)
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(boxes[i, 2], boxes[j, 2]) - max(boxes[i, 0], boxes[j, 0])
                oy = min(boxes[i, 3], boxes[j, 3]) - max(boxes[i, 1], boxes[j, 1])
                if ox > 0.001 and oy > 0.001:
                    wi = boxes[i, 2] - boxes[i, 0]; wj = boxes[j, 2] - boxes[j, 0]
                    hi = boxes[i, 3] - boxes[i, 1]; hj = boxes[j, 3] - boxes[j, 1]
                    cxi, cxj = (boxes[i, 0] + boxes[i, 2]) / 2.0, (boxes[j, 0] + boxes[j, 2]) / 2.0
                    cyi, cyj = (boxes[i, 1] + boxes[i, 3]) / 2.0, (boxes[j, 1] + boxes[j, 3]) / 2.0
                    dx = abs(cxj - cxi) / max(wi + wj, 1.0)
                    dy = abs(cyj - cyi) / max(hi + hj, 1.0)
                    if dx >= dy:
                        sh = (ox + padding) / 2.0
                        if cxi <= cxj:
                            boxes[i, [0, 2]] -= sh; boxes[j, [0, 2]] += sh
                        else:
                            boxes[i, [0, 2]] += sh; boxes[j, [0, 2]] -= sh
                    else:
                        sh = (oy + padding) / 2.0
                        if cyi <= cyj:
                            boxes[i, [1, 3]] -= sh; boxes[j, [1, 3]] += sh
                        else:
                            boxes[i, [1, 3]] += sh; boxes[j, [1, 3]] -= sh
    return boxes


def build_candidate(G_in: nx.Graph, rings: List[np.ndarray]) -> Candidate:
    """Rings (canvas units, centred) -> validated Candidate with a NetworkX floorplan graph."""
    raw_polys = [_to_polygon(r) for r in rings]
    good = [p for p in raw_polys if p is not None and not p.is_empty]
    if good:
        x0 = min(p.bounds[0] for p in good)
        y0 = min(p.bounds[1] for p in good)
        raw_polys = [affinity.translate(p, 8 - x0, 8 - y0) if p is not None else None for p in raw_polys]

    node_names = list(G_in.nodes)
    node_types = [G_in.nodes[n]["type"] for n in node_names]
    req_areas = [G_in.nodes[n]["area"] for n in node_names]
    n = len(node_names)

    liv_idx = next((i for i, t in enumerate(node_types) if t == "living"), -1)

    # 1. Initialize rooms with requested areas and predicted aspect ratios
    boxes = []
    for i in range(n):
        p = raw_polys[i]
        t = node_types[i]
        req = max(req_areas[i], MIN_ROOM_AREAS.get(t, 30.0))
        if p is not None and not p.is_empty:
            cx, cy = p.centroid.x, p.centroid.y
            bx0, by0, bx1, by1 = p.bounds
            pw = max(bx1 - bx0, 10.0)
            ph = max(by1 - by0, 10.0)
            ar = float(np.clip(pw / ph, 1 / 2.8, 2.8))
        else:
            cx, cy = 128.0, 128.0
            ar = 1.0
        w = np.sqrt(req * ar)
        h = req / w
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    boxes = np.array(boxes, dtype=np.float64)

    # 2. Relax all boxes to completely eliminate overlaps while preserving exact areas and topology
    active_indices = [i for i in range(n) if node_types[i] not in OVERLAP_EXEMPT_TYPES]
    active_names = [node_names[i] for i in active_indices]
    if len(active_indices) > 1:
        sub_boxes = boxes[active_indices]
        rel_boxes = relax_box_collisions(sub_boxes, G_in=G_in, node_names=active_names, max_iters=80, padding=0.05)
        boxes[active_indices] = rel_boxes

    polys = [box(*b) for b in boxes]
    for i in range(n):
        if node_types[i] in OVERLAP_EXEMPT_TYPES:
            p = raw_polys[i]
            req = req_areas[i]
            if p is not None and not p.is_empty:
                cx, cy = p.centroid.x, p.centroid.y
                w = max(np.sqrt(req), 4.0)
                h = req / w
                polys[i] = box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    # 3. Living Room Geometry: Strictly ensure 4 to 8 vertices
    if liv_idx >= 0 and polys[liv_idx] is not None:
        liv_geom = polys[liv_idx]
        outer_indices = [i for i in active_indices if i != liv_idx]
        inter_outers = []
        for i in outer_indices:
            if liv_geom.intersects(polys[i]):
                ia = liv_geom.intersection(polys[i]).area
                if ia > 10.0:
                    inter_outers.append((ia, i))
        inter_outers.sort(reverse=True)
        cutouts_done = set()
        for _, idx in inter_outers:
            if len(cutouts_done) >= 2:
                break
            diff = liv_geom.difference(polys[idx])
            if diff.is_valid and diff.geom_type == "Polygon":
                nv = len(diff.exterior.coords) - 1
                if 4 <= nv <= 8 and abs(diff.area - req_areas[liv_idx]) / max(req_areas[liv_idx], 1.0) <= 0.25:
                    liv_geom = diff
                    cutouts_done.add(idx)

        # Separate any outer room that still overlaps with the living room
        for i in outer_indices:
            if i not in cutouts_done and liv_geom.intersects(polys[i]):
                inter = liv_geom.intersection(polys[i])
                if inter.area > 0.001:
                    ox = min(boxes[i][2], liv_geom.bounds[2]) - max(boxes[i][0], liv_geom.bounds[0])
                    oy = min(boxes[i][3], liv_geom.bounds[3]) - max(boxes[i][1], liv_geom.bounds[1])
                    if ox > 0 and oy > 0:
                        lcx, lcy = liv_geom.centroid.x, liv_geom.centroid.y
                        bcx, bcy = (boxes[i][0] + boxes[i][2]) / 2.0, (boxes[i][1] + boxes[i][3]) / 2.0
                        if abs(bcx - lcx) >= abs(bcy - lcy):
                            sh = ox + 0.05
                            if bcx >= lcx:
                                boxes[i][0] += sh; boxes[i][2] += sh
                            else:
                                boxes[i][0] -= sh; boxes[i][2] -= sh
                        else:
                            sh = oy + 0.05
                            if bcy >= lcy:
                                boxes[i][1] += sh; boxes[i][3] += sh
                            else:
                                boxes[i][1] -= sh; boxes[i][3] -= sh
                        polys[i] = box(*boxes[i])

        polys[liv_idx] = liv_geom

    # 4. Final safety pass on outer boxes to ensure zero overlap between outer rooms
    outer_indices = [i for i in active_indices if i != liv_idx]
    if len(outer_indices) > 1:
        o_sub = boxes[outer_indices]
        o_rel = relax_box_collisions(o_sub, max_iters=40, padding=0.05)
        boxes[outer_indices] = o_rel
        for idx, b in zip(outer_indices, o_rel):
            polys[idx] = box(*b)

    # 5. Guaranteed zero-overlap safety net (any remaining tiny overlap)
    for _ in range(15):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = polys[i], polys[j]
                if p1 is None or p2 is None or p1.is_empty or p2.is_empty:
                    continue
                t1, t2 = node_types[i], node_types[j]
                if t1 in OVERLAP_EXEMPT_TYPES or t2 in OVERLAP_EXEMPT_TYPES:
                    continue
                if p1.intersects(p2):
                    inter = p1.intersection(p2)
                    if inter.area > 0.001:
                        changed = True
                        b1 = boxes[i] if t1 != "living" else p1.bounds
                        b2 = boxes[j] if t2 != "living" else p2.bounds
                        ox = min(b1[2], b2[2]) - max(b1[0], b2[0])
                        oy = min(b1[3], b2[3]) - max(b1[1], b2[1])
                        cxi = (b1[0] + b1[2]) / 2.0; cxj = (b2[0] + b2[2]) / 2.0
                        cyi = (b1[1] + b1[3]) / 2.0; cyj = (b2[1] + b2[3]) / 2.0
                        
                        if t1 == "living" and t2 != "living":
                            if ox <= oy:
                                sh = ox + 0.05
                                if cxi <= cxj: boxes[j][0] += sh; boxes[j][2] += sh
                                else: boxes[j][0] -= sh; boxes[j][2] -= sh
                            else:
                                sh = oy + 0.05
                                if cyi <= cyj: boxes[j][1] += sh; boxes[j][3] += sh
                                else: boxes[j][1] -= sh; boxes[j][3] -= sh
                            polys[j] = box(*boxes[j])
                        elif t2 == "living" and t1 != "living":
                            if ox <= oy:
                                sh = ox + 0.05
                                if cxi <= cxj: boxes[i][0] -= sh; boxes[i][2] -= sh
                                else: boxes[i][0] += sh; boxes[i][2] += sh
                            else:
                                sh = oy + 0.05
                                if cyi <= cyj: boxes[i][1] -= sh; boxes[i][3] -= sh
                                else: boxes[i][1] += sh; boxes[i][3] += sh
                            polys[i] = box(*boxes[i])
                        else:
                            if ox <= oy:
                                sh = (ox + 0.05) / 2.0
                                if cxi <= cxj:
                                    boxes[i][0] -= sh; boxes[i][2] -= sh; boxes[j][0] += sh; boxes[j][2] += sh
                                else:
                                    boxes[i][0] += sh; boxes[i][2] += sh; boxes[j][0] -= sh; boxes[j][2] -= sh
                            else:
                                sh = (oy + 0.05) / 2.0
                                if cyi <= cyj:
                                    boxes[i][1] -= sh; boxes[i][3] -= sh; boxes[j][1] += sh; boxes[j][3] += sh
                                else:
                                    boxes[i][1] += sh; boxes[i][3] += sh; boxes[j][1] -= sh; boxes[j][3] -= sh
                            polys[i] = box(*boxes[i])
                            polys[j] = box(*boxes[j])
        if not changed:
            break

    polys = polys
    # ----------------------------------------------

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
            # Create walls along all boundaries (interior + exterior)
            boundaries = unary_union([g.boundary for g in allg if g.is_valid])
            wall = boundaries.buffer(1.5, cap_style=2, join_style=2)
            
            # Subtract doors from the wall to create clear openings
            if "door" in plan and not plan["door"].is_empty:
                wall = wall.difference(plan["door"])
                
            if not wall.is_empty:
                plan["wall"] = wall
        except Exception:
            pass
            
    return plan


# ----------------------------------------------------------------------------
# sampling pipeline (§20-23)
def align_walls(rings: List[np.ndarray], tol: float = 1.5) -> List[np.ndarray]:
    """Snap coordinates across all rooms to shared grid lines if they are within `tol`."""
    if not rings: return rings
    out = [r.copy() for r in rings]
    for axis in (0, 1):
        vals = []
        for r in out:
            if len(r) >= 3: vals.extend(r[:, axis])
        if not vals: continue
        vals = np.array(vals)
        order = np.argsort(vals)
        vals_sorted = vals[order]
        diffs = np.diff(vals_sorted)
        breaks = np.where(diffs > tol)[0] + 1
        groups = np.split(order, breaks)
        means = [vals[g].mean() for g in groups]
        for g, m in zip(groups, means):
            vals[g] = m
        
        idx = 0
        for r in out:
            if len(r) >= 3:
                r[:, axis] = vals[idx:idx + len(r)]
                idx += len(r)
    return out

# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_floorplans(model, graphs: List[nx.Graph], K: int = 8, steps: int = 50, eta: float = 0.0,
                        device: str = "cpu", seed: int = 0, batch_size: int = 32,
                        sample_counts: bool = False, drop_invalid: bool = False,
                        snap_grid: float = 0.0, guidance: float = 0.0,
                        guide_start: float = 0.6, guide_iters: int = 2) -> List[List[Candidate]]:
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
        out = model.sample(batch, steps=steps, eta=eta, sample_counts=sample_counts, generator=gen,
                           guidance=guidance, guide_start=guide_start, guide_iters=guide_iters)
        P, vm = model.to_polygons(out["x"], out["tok_mask"])

        # --- MULTI-OBJECTIVE ACTIVE CONTOUR TTO ---
        with torch.enable_grad():
            from .losses import (ortho_loss, area_loss, validity_loss, aspect_ratio_loss,
                                 overlap_loss, connectivity_losses)
            from .config import ROOM_TYPE_TO_ID
            P_opt = P.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([P_opt], lr=0.08)
            rmask = batch["room_mask"]
            target_area = batch["area"]
            vm_r = vm & rmask[..., None]
            exempt = batch["room_type"] == ROOM_TYPE_TO_ID["front_door"]
            
            for step in range(50):
                lr = 0.08 * (1.0 - 0.7 * step / 50)
                for pg in opt.param_groups: pg["lr"] = lr
                opt.zero_grad()
                ol = ortho_loss(P_opt, vm_r, rmask).sum()
                al = area_loss(P_opt, vm_r, rmask, target_area).sum()
                vl = validity_loss(P_opt, vm_r, rmask, target_area).sum()
                aspl = aspect_ratio_loss(P_opt, vm_r, rmask, max_ratio=3.0).sum()
                ovl = overlap_loss(P_opt, vm_r, rmask, grid=32, exempt=exempt).sum()
                conn, nonc = connectivity_losses(P_opt, vm_r, rmask, batch["adj"])
                loss = (3.0 * ol + 6.0 * al + 1.0 * vl + 2.0 * aspl + 
                        8.0 * ovl + 30.0 * conn.sum() + 10.0 * nonc.sum())
                loss.backward()
                opt.step()
            P = P_opt.detach()
        # ------------------------------------------
        P, vm = P.cpu().numpy().astype(np.float64), vm.cpu().numpy()
        for b, (gi, G) in enumerate(chunk):
            n = len(G)
            rings = []
            for i in range(n):
                r = P[b, i][vm[b, i]]
                if snap_grid > 0:
                    r = np.round(r / snap_grid) * snap_grid
                rings.append(r)
            rings = align_walls(rings)
            rings = [clean_ring(r) if len(r) >= 3 else r for r in rings]
            results[gi].append(build_candidate(G, rings))
    for gi in range(len(results)):
        cands = results[gi]
        if drop_invalid:
            cands = [c for c in cands if c.valid]
        results[gi] = sorted(cands, key=lambda c: (not c.valid, c.score))
    return results
