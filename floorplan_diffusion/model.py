"""Graph Transformer encoder + conditional coordinate denoiser (plan §8-14).

Graph encoder input is strictly (room type, area, typed edges): ground-truth
coordinates never reach it (Rule 8).  The denoiser is conditioned on the
resulting room embeddings through (a) per-token injection of the room's own
embedding and (b) cross-attention to all room embeddings + the global token.
"""
from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, NUM_EDGE_TYPES, NUM_ROOM_TYPES

REL_NONE, REL_GLOBAL, REL_SELF = 0, NUM_EDGE_TYPES + 1, NUM_EDGE_TYPES + 2
NUM_REL = NUM_EDGE_TYPES + 3


class MHA(nn.Module):
    def __init__(self, d: int, heads: int, kv_dim: int | None = None):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(kv_dim or d, d)
        self.v = nn.Linear(kv_dim or d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x, kv, attn_mask=None):
        B, L, _ = x.shape
        S = kv.shape[1]
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(kv).view(B, S, self.h, self.dk).transpose(1, 2)
        v = self.v(kv).view(B, S, self.h, self.dk).transpose(1, 2)
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)        # keep masks dtype-consistent under autocast
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(y.transpose(1, 2).reshape(B, L, -1))


def _mlp(d, mult=4, drop=0.0):
    return nn.Sequential(nn.Linear(d, d * mult), nn.GELU(), nn.Dropout(drop), nn.Linear(d * mult, d))


# ----------------------------------------------------------------------------
# Graph Transformer
# ----------------------------------------------------------------------------

class GraphBlock(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MHA(d, heads)
        self.mlp = _mlp(d, drop=drop)
        self.rel_bias = nn.Embedding(NUM_REL, heads)      # edge-type-aware attention bias
        self.drop = nn.Dropout(drop)

    def forward(self, x, rel, key_ok):
        bias = self.rel_bias(rel).permute(0, 3, 1, 2)      # B,H,L,L
        bias = bias.masked_fill(~key_ok[:, None, None, :], float("-inf"))
        x = x + self.drop(self.attn(self.n1(x), self.n1(x), bias.to(x.dtype)))
        return x + self.drop(self.mlp(self.n2(x)))


class GraphTransformer(nn.Module):
    """Graph -> per-room embeddings h_i and a global floorplan embedding."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.type_emb = nn.Embedding(NUM_ROOM_TYPES, d)            # §9 categorical
        self.area_mlp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))   # §9 numerical
        self.deg_proj = nn.Linear(NUM_EDGE_TYPES, d)               # typed degree features
        self.global_tok = nn.Parameter(torch.randn(d) * 0.02)      # §11 global token
        self.blocks = nn.ModuleList([GraphBlock(d, cfg.n_heads, cfg.dropout)
                                     for _ in range(cfg.graph_layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, room_type, area_feat, adj, room_mask):
        B, N = room_type.shape
        deg = torch.stack([(adj == e).sum(-1) for e in range(1, NUM_EDGE_TYPES + 1)], -1).float()
        x = self.type_emb(room_type) + self.area_mlp(area_feat) + self.deg_proj(deg / 4.0)
        x = torch.cat([self.global_tok.expand(B, 1, -1), x], dim=1)       # B,N+1,d
        rel = torch.full((B, N + 1, N + 1), REL_NONE, dtype=torch.long, device=x.device)
        rel[:, 1:, 1:] = adj
        rel[:, 0, :] = REL_GLOBAL
        rel[:, :, 0] = REL_GLOBAL
        eye = torch.eye(N + 1, dtype=torch.bool, device=x.device)
        rel = rel.masked_fill(eye, REL_SELF)
        key_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), room_mask], 1)
        for blk in self.blocks:
            x = blk(x, rel, key_ok)
        x = self.norm(x)
        return x[:, 1:], x[:, 0]


# ----------------------------------------------------------------------------
# Denoiser
# ----------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class DenoiseBlock(nn.Module):
    """self-attn over all tokens -> cross-attn to room embeddings -> MLP, all AdaLN-modulated."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d, elementwise_affine=False) for _ in range(3))
        self.self_attn = MHA(d, heads)
        self.cross_attn = MHA(d, heads)
        self.mlp = _mlp(d, drop=drop)
        self.mod = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, mem, self_mask, cross_mask):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.mod(cond)[:, None].chunk(9, dim=-1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * self.self_attn(h, h, self_mask)
        h = self.n2(x) * (1 + s2) + b2
        x = x + g2 * self.cross_attn(h, mem, cross_mask)
        h = self.n3(x) * (1 + s3) + b3
        return x + g3 * self.mlp(h)


class FloorplanDenoiser(nn.Module):
    """Graph Transformer + vertex/polygon Transformer predicting clean coordinates x0."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, K, D = cfg.d_model, cfg.tokens_per_room, cfg.coord_dim
        self.graph = GraphTransformer(cfg)
        self.in_proj = nn.Linear(D, d)
        self.vert_emb = nn.Embedding(K, d)                         # vertex position/index embedding
        self.room_proj = nn.Linear(d, d)
        self.time_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.glob_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList([DenoiseBlock(d, cfg.n_heads, cfg.dropout)
                                     for _ in range(cfg.denoiser_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D)
        # vertex-count prediction head (§6): classes 0..V, only 3..V are valid
        self.count_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.max_vertices + 1))

    # -- conditioning ---------------------------------------------------------
    def encode(self, room_type, area_feat, adj, room_mask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        room_h, glob_h = self.graph(room_type, area_feat, adj, room_mask)
        return room_h, glob_h, self.count_head(room_h)

    # -- denoise --------------------------------------------------------------
    def denoise(self, x_t, t, room_h, glob_h, room_mask, tok_mask):
        """x_t: [B,N,K,D] noisy coords; tok_mask: [B,N,K] valid tokens. Returns x0_hat [B,N,K,D]."""
        B, N, K, D = x_t.shape
        d = room_h.shape[-1]
        cond = self.time_mlp(timestep_embedding(t, d)) + self.glob_proj(glob_h)
        pos = self.vert_emb(torch.arange(K, device=x_t.device))               # K,d
        tok = self.in_proj(x_t) + pos[None, None] + self.room_proj(room_h)[:, :, None]
        tok = tok.reshape(B, N * K, d)
        ok = (tok_mask & room_mask[:, :, None]).reshape(B, N * K)
        tok = tok * ok.unsqueeze(-1)
        # additive key-padding masks; every query always has >=1 valid key (global token / real tokens)
        self_mask = torch.zeros(B, 1, 1, N * K, device=x_t.device, dtype=tok.dtype)
        self_mask = self_mask.masked_fill(~ok[:, None, None, :], float("-inf"))
        mem = torch.cat([glob_h[:, None], room_h], dim=1)
        mem_ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x_t.device), room_mask], 1)
        cross_mask = torch.zeros(B, 1, 1, N + 1, device=x_t.device, dtype=tok.dtype)
        cross_mask = cross_mask.masked_fill(~mem_ok[:, None, None, :], float("-inf"))
        for blk in self.blocks:
            tok = blk(tok, cond, mem, self_mask, cross_mask)
        out = self.out(self.out_norm(tok)).view(B, N, K, D)
        return out * (tok_mask & room_mask[:, :, None]).unsqueeze(-1)
