"""Differentiable geometric approximations used during training (plan §18-19).

Shapely is *not* used here.  All functions take polygons as padded vertex
tensors in **canvas units**:

    P     [B, N, V, 2]   ordered vertices (CCW expected)
    vmask [B, N, V]      valid-vertex mask (already includes room validity)
    rmask [B, N]         valid-room mask

and return one scalar per batch element, [B].
"""
from __future__ import annotations
import math
from typing import Dict

import torch
import torch.nn.functional as F

from .config import (CONNECT_TOL, EDGE_TYPES, NONEDGE_MARGIN, NUM_EDGE_TYPES)

BIG = 1e4


def rect_to_polygon(rect: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) [...,4] -> CCW corners [...,4,2] starting at the lowest-left corner."""
    cx, cy, w, h = rect.unbind(-1)
    w, h = w.abs(), h.abs()
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return torch.stack([torch.stack([x0, y0], -1), torch.stack([x1, y0], -1),
                        torch.stack([x1, y1], -1), torch.stack([x0, y1], -1)], dim=-2)


def _next_idx(vmask: torch.Tensor) -> torch.Tensor:
    """Index of the successor vertex (wraps to 0 after the last valid vertex)."""
    V = vmask.shape[-1]
    n = vmask.sum(-1, keepdim=True).clamp(min=1)
    i = torch.arange(V, device=vmask.device).expand_as(vmask)
    nxt = i + 1
    return torch.where(nxt >= n, torch.zeros_like(nxt), nxt)


def _succ(P, vmask):
    idx = _next_idx(vmask).unsqueeze(-1).expand(-1, -1, -1, 2)
    return torch.gather(P, 2, idx)


def signed_area(P, vmask):
    """Shoelace signed area [B,N] (positive = CCW)."""
    Q = _succ(P, vmask)
    cross = P[..., 0] * Q[..., 1] - Q[..., 0] * P[..., 1]
    return 0.5 * (cross * vmask).sum(-1)


# ----------------------------------------------------------------------------
def area_loss(P, vmask, rmask, target_area):
    """Predicted |area| vs requested area, compared on a sqrt (length) scale."""
    A = signed_area(P, vmask).abs()
    ratio = (A + 1.0).sqrt() / target_area.clamp(min=1.0).sqrt() - 1.0
    l = F.smooth_l1_loss(ratio, torch.zeros_like(ratio), reduction="none", beta=0.2)
    return (l * rmask).sum(-1) / rmask.sum(-1).clamp(min=1)


def _orient(a, b, c):
    """Signed distance of c to the line a->b (in length units)."""
    ab, ac = b - a, c - a
    cross = ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0]
    return cross / (ab.norm(dim=-1) + 1e-3)


def validity_loss(P, vmask, rmask, target_area):
    """Degenerate / clockwise polygons and soft self-intersections."""
    A = signed_area(P, vmask)
    degenerate = F.relu(0.1 * target_area - A) / target_area.clamp(min=1.0)
    degenerate = (degenerate * rmask).sum(-1) / rmask.sum(-1).clamp(min=1)

    B, N, V, _ = P.shape
    Q = _succ(P, vmask)
    a_i, b_i = P[:, :, :, None], Q[:, :, :, None]       # edge i
    a_j, b_j = P[:, :, None, :], Q[:, :, None, :]       # edge j
    s1 = torch.tanh(_orient(a_i, b_i, a_j))
    s2 = torch.tanh(_orient(a_i, b_i, b_j))
    s3 = torch.tanh(_orient(a_j, b_j, a_i))
    s4 = torch.tanh(_orient(a_j, b_j, b_i))
    cross = F.relu(-s1 * s2) * F.relu(-s3 * s4)          # [B,N,V,V]
    n = vmask.sum(-1)[:, :, None, None]
    ii = torch.arange(V, device=P.device)
    di = (ii[:, None] - ii[None, :]).abs()
    nonadj = (di > 1)[None, None] & ~(di[None, None] == (n - 1))
    valid = vmask[:, :, :, None] & vmask[:, :, None, :] & nonadj
    inter = (cross * valid).sum((-1, -2)) / 2.0           # [B,N]
    inter = (inter * rmask).sum(-1) / rmask.sum(-1).clamp(min=1)
    return degenerate + inter


def soft_inside(P, vmask, pts):
    """Smooth point-in-polygon indicator via normalised winding number.

    P [B,N,V,2], pts [B,G,2] -> [B,N,G] in [0,1] (CCW polygons).
    """
    Q = _succ(P, vmask)
    a = P[:, :, None] - pts[:, None, :, None]            # B,N,G,V,2
    b = Q[:, :, None] - pts[:, None, :, None]
    ang = torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
                      (a * b).sum(-1))
    w = (ang * vmask[:, :, None, :]).sum(-1) / (2 * math.pi)
    return w.clamp(0.0, 1.0)


def overlap_loss(P, vmask, rmask, grid: int = 64, exempt=None):
    """Soft rasterised pairwise overlap area, relative to total room area.

    ``exempt`` [B,N] marks rooms (e.g. front_door) whose overlaps are ignored."""
    B, N, V, _ = P.shape
    big = 1e6
    m = vmask.unsqueeze(-1)
    lo = torch.where(m, P, torch.full_like(P, big)).amin(dim=(1, 2)).detach()
    hi = torch.where(m, P, torch.full_like(P, -big)).amax(dim=(1, 2)).detach()
    lin = (torch.arange(grid, device=P.device) + 0.5) / grid              # cell centres in [0,1]
    gx = lo[:, 0:1] + (hi - lo)[:, 0:1] * lin[None]                       # B,grid
    gy = lo[:, 1:2] + (hi - lo)[:, 1:2] * lin[None]
    xs = gx[:, :, None].expand(B, grid, grid)
    ys = gy[:, None, :].expand(B, grid, grid)
    pts = torch.stack([xs, ys], -1).reshape(B, grid * grid, 2)
    cell = ((hi - lo)[:, 0] / grid) * ((hi - lo)[:, 1] / grid)           # [B]
    keep = rmask if exempt is None else (rmask & ~exempt)
    ins = soft_inside(P, vmask, pts) * keep.unsqueeze(-1)                 # B,N,G
    inter = torch.einsum("big,bjg->bij", ins, ins)                        # B,N,N
    eye = torch.eye(N, dtype=torch.bool, device=P.device)[None]
    inter = inter.masked_fill(eye, 0.0).sum((-1, -2)) / 2.0 * cell
    total = signed_area(P, vmask).abs().mul(rmask).sum(-1).clamp(min=1.0).detach()
    return inter / total


def pair_distances(P, vmask):
    """Minimum boundary distance between every pair of rooms [B,N,N] (canvas units)."""
    Q = _succ(P, vmask)
    p = P[:, :, None, :, None]                 # B,N,1,V,1,2   vertex k of room i
    a = P[:, None, :, None, :]                 # B,1,N,1,V,2   segment l of room j
    b = Q[:, None, :, None, :]
    ab = b - a
    t = ((p - a) * ab).sum(-1) / ((ab * ab).sum(-1) + 1e-6)
    t = t.clamp(0.0, 1.0)
    d = (p - (a + t[..., None] * ab)).norm(dim=-1)                       # B,N,N,V,V
    ok = vmask[:, :, None, :, None] & vmask[:, None, :, None, :]
    d = d.masked_fill(~ok, BIG)
    M = d.amin(dim=(-1, -2))                                              # B,N,N
    return torch.minimum(M, M.transpose(1, 2))


def connectivity_losses(P, vmask, rmask, adj):
    """(connectivity, non-connectivity) losses from typed adjacency ``adj`` [B,N,N] (0 = none)."""
    D = pair_distances(P, vmask)
    N = P.shape[1]
    tol_tab = torch.zeros(NUM_EDGE_TYPES + 1, device=P.device)
    for i, e in enumerate(EDGE_TYPES):
        tol_tab[i + 1] = CONNECT_TOL[e]
    tol = tol_tab[adj]
    pair_ok = rmask[:, :, None] & rmask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=P.device)[None]
    is_edge = (adj > 0) & pair_ok
    non_edge = (adj == 0) & pair_ok
    conn = F.smooth_l1_loss(F.relu(D - tol) / 10.0, torch.zeros_like(D), reduction="none")
    nonc = F.relu(NONEDGE_MARGIN - D) / 10.0
    conn = (conn * is_edge).sum((-1, -2)) / is_edge.sum((-1, -2)).clamp(min=1)
    nonc = (nonc * non_edge).sum((-1, -2)) / non_edge.sum((-1, -2)).clamp(min=1)
    return conn, nonc


def ortho_loss(P, vmask, rmask):
    """Encourages edges to be perfectly horizontal or vertical (axis-aligned)."""
    Q = _succ(P, vmask)
    edge = (Q - P).abs()
    # min(dx, dy) should be 0 for axis-aligned edges
    ortho = torch.minimum(edge[..., 0], edge[..., 1])
    ortho = (ortho * vmask).sum(-1) / vmask.sum(-1).clamp(min=1)
    return (ortho * rmask).sum(-1) / rmask.sum(-1).clamp(min=1)


def strict_gap_loss(P, vmask, rmask, adj):
    """Strictly penalizes distance between connected rooms to pull them flush, overriding wall thickness."""
    D = pair_distances(P, vmask)
    N = P.shape[1]
    pair_ok = rmask[:, :, None] & rmask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=P.device)[None]
    is_edge = (adj > 0) & pair_ok
    gap = (D * is_edge).sum((-1, -2)) / is_edge.sum((-1, -2)).clamp(min=1)
    return gap


def aux_losses(P, vmask, rmask, target_area, adj, exempt=None) -> Dict[str, torch.Tensor]:
    conn, nonc = connectivity_losses(P, vmask, rmask, adj)
    return dict(area=area_loss(P, vmask, rmask, target_area),
                valid=validity_loss(P, vmask, rmask, target_area),
                overlap=overlap_loss(P, vmask, rmask, exempt=exempt),
                conn=conn, nonconn=nonc,
                ortho=ortho_loss(P, vmask, rmask),
                gap=strict_gap_loss(P, vmask, rmask, adj))
