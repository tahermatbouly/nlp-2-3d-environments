"""
model.py — Graph-Conditioned CVAE for floor plan generation.

Architecture
------------
1.  GNN Encoder  (GATConv × 3 layers with edge_attr + residual connections)
        room features  →  per-node embeddings (128-d)

2.  VAE Encoder  (MLP, training only)
        node embeddings + real bounding boxes  →  μ, log σ²  →  sample z

3.  VAE Decoder  (MLP)
        node embeddings + z  →  predicted bounding boxes (x, y, w, h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

import config as cfg


class FloorPlanCVAE(nn.Module):

    def __init__(self):
        super().__init__()

        in_dim = cfg.NODE_FEATURE_DIM   # 9
        h_dim  = cfg.GNN_HIDDEN_DIM     # 128
        z_dim  = cfg.LATENT_DIM         # 32
        e_dim  = cfg.NUM_EDGE_TYPES     # 4
        drop   = cfg.DROPOUT            # 0.1

        # ── Part 1: GNN encoder (GATConv with edge_attr support) ────────
        self.gnn1 = GATConv(in_dim, h_dim, heads=2, concat=False,
                            edge_dim=e_dim, dropout=drop)
        self.gnn2 = GATConv(h_dim,  h_dim, heads=2, concat=False,
                            edge_dim=e_dim, dropout=drop)
        self.gnn3 = GATConv(h_dim,  h_dim, heads=2, concat=False,
                            edge_dim=e_dim, dropout=drop)
        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.norm3 = nn.LayerNorm(h_dim)
        # projection for residual from input dim → h_dim
        self.proj_in = nn.Linear(in_dim, h_dim) if in_dim != h_dim else nn.Identity()

        # ── Part 2: VAE encoder  (used during training only) ────────────
        #     input  = node embedding (128) + ground-truth bbox (4)
        #     output = μ (32) and log σ² (32)
        self.vae_enc = nn.Sequential(
            nn.Linear(h_dim + 4, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.mu_head     = nn.Linear(128, z_dim)
        self.logvar_head = nn.Linear(128, z_dim)

        # ── Part 3: VAE decoder ─────────────────────────────────────────
        #     input  = node embedding (128) + z (32)
        #     output = predicted bbox (4): cx, cy, w, h  in [0, 1]
        self.decoder = nn.Sequential(
            nn.Linear(h_dim + z_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 4),
            nn.Sigmoid(),       # keeps output in [0, 1]
        )

    # ── helpers ─────────────────────────────────────────────────────────

    def _encode_graph(self, x, edge_index, edge_attr=None):
        """Run 3 GATConv layers with residual connections → per-node embeddings."""
        h = F.relu(self.norm1(self.gnn1(x, edge_index, edge_attr=edge_attr)))
        h = F.relu(self.norm2(self.gnn2(h, edge_index, edge_attr=edge_attr))) + h   # residual
        h = F.relu(self.norm3(self.gnn3(h, edge_index, edge_attr=edge_attr))) + h   # residual
        return h                                          # (N, 128)

    def _encode_vae(self, node_emb, bboxes):
        """Compress (node embeddings + real bboxes) into μ and log σ²."""
        h = self.vae_enc(torch.cat([node_emb, bboxes], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def _reparameterise(mu, logvar):
        """Sample z = μ + σ ⊙ ε   (ε ~ N(0, I))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _decode(self, node_emb, z):
        """Decode (node embeddings + z) → predicted bounding boxes."""
        return self.decoder(torch.cat([node_emb, z], dim=-1))

    # ── public API ──────────────────────────────────────────────────────

    def forward(self, data):
        """Training forward pass.

        Returns
        -------
        pred_bboxes : (N, 4)
        mu          : (N, 32)
        logvar      : (N, 32)
        """
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        node_emb = self._encode_graph(data.x, data.edge_index, edge_attr)
        mu, logvar = self._encode_vae(node_emb, data.y)
        z = self._reparameterise(mu, logvar)
        pred_bboxes = self._decode(node_emb, z)
        return pred_bboxes, mu, logvar

    @torch.no_grad()
    def generate(self, data):
        """Inference: sample z from N(0, I) and decode.

        Returns
        -------
        pred_bboxes : (N, 4)
        """
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
        node_emb = self._encode_graph(data.x, data.edge_index, edge_attr)
        z = torch.randn(
            node_emb.size(0), cfg.LATENT_DIM, device=node_emb.device
        )
        return self._decode(node_emb, z)
