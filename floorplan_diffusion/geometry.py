"""Polygon preprocessing (DIFFUSION_PLAN.md §5).

Pipeline: validate -> remove duplicate points -> remove redundant collinear
points -> fix orientation (CCW) -> fix starting vertex -> ordered coordinates.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

from .config import COLLINEAR_TOL, MAX_VERTICES, MIN_VERTICES


def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
        if not polys:
            return None
        return max(polys, key=lambda p: p.area)
    return None


def fix_polygon(geom) -> Optional[Polygon]:
    """Rule 1: return a valid, hole-free Polygon or None if it cannot be fixed."""
    poly = _largest_polygon(geom)
    if poly is None:
        return None
    if not poly.is_valid:
        poly = _largest_polygon(make_valid(poly))
        if poly is None:
            return None
    poly = Polygon(poly.exterior)          # rooms are simple regions, drop holes
    if not poly.is_valid:
        poly = _largest_polygon(poly.buffer(0))
    if poly is None or poly.is_empty or poly.area <= 0:
        return None
    return poly


def clean_ring(coords: np.ndarray, tol: float = COLLINEAR_TOL) -> np.ndarray:
    """Rules 2: drop duplicate vertices and vertices lying on a straight edge.

    ``coords`` is an (n, 2) *open* ring (first point not repeated).
    """
    pts = [tuple(p) for p in coords]
    # duplicates (consecutive, including wrap-around)
    dedup = []
    for p in pts:
        if not dedup or np.hypot(p[0] - dedup[-1][0], p[1] - dedup[-1][1]) > 1e-9:
            dedup.append(p)
    while len(dedup) > 1 and np.hypot(dedup[0][0] - dedup[-1][0], dedup[0][1] - dedup[-1][1]) <= 1e-9:
        dedup.pop()
    pts = dedup
    changed = True
    while changed and len(pts) > 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % n])
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) <= tol:                # b lies on segment a-c (or is a spike)
                pts.pop(i)
                changed = True
                break
    return np.asarray(pts, dtype=np.float64)


def signed_area(coords: np.ndarray) -> float:
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def canonical_start(coords: np.ndarray) -> np.ndarray:
    """Rule 4: start at the lowest vertex (min y, ties within 1e-3 -> min x)."""
    ymin = coords[:, 1].min()
    cand = np.where(coords[:, 1] <= ymin + 1e-3)[0]          # tolerate float noise in "equal" y (1e-3 units)
    idx = cand[np.argmin(coords[cand, 0])]
    return np.roll(coords, -idx, axis=0)


def preprocess_polygon(geom, max_vertices: int = MAX_VERTICES) -> Optional[np.ndarray]:
    """Full pipeline. Returns an (n, 2) float64 open ring or None if unusable.

    Polygons with more than ``max_vertices`` vertices are progressively
    simplified (Douglas-Peucker); if that still fails the polygon is rejected.
    """
    poly = fix_polygon(geom)
    if poly is None:
        return None
    tol = 0.0
    for _ in range(12):
        cand = poly if tol == 0.0 else _largest_polygon(poly.simplify(tol, preserve_topology=True))
        if cand is None or not cand.is_valid:
            tol = tol * 2 if tol else 0.1
            continue
        ring = np.asarray(cand.exterior.coords)[:-1]
        ring = clean_ring(ring)
        if len(ring) >= MIN_VERTICES and len(ring) <= max_vertices:
            p = Polygon(ring)
            if not p.is_valid or p.area <= 0:
                return None
            ring = orient(p, sign=1.0).exterior.coords  # CCW
            ring = np.asarray(ring)[:-1]
            return canonical_start(ring)
        tol = tol * 2 if tol else 0.1
    return None


def ring_to_polygon(ring: np.ndarray) -> Optional[Polygon]:
    """Shapely polygon from an ordered ring (None if fewer than 3 distinct points)."""
    if ring is None or len(ring) < 3:
        return None
    return Polygon(ring)
