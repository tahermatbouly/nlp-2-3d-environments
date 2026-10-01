"""
model.py — Diffusion model for floorplan generation with graph conditioning.

Architecture
------------
1.  Graph Encoder (GATConv × 3 layers with edge_attr + residual connections)
        room features  →  per-node embeddings (256-d) → global graph embedding (256-d)

2.  UNet with AdaIN conditioning
        Takes noisy image [3, H, W] and timestep t, predicts noise.
        Conditioned on global graph embedding via adaptive instance normalization.

3.  Forward diffusion: gradually add noise to the image.
    Reverse diffusion: denoise network predicts noise to remove.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv

from pipeline import config as cfg


# ── Helper Layers ───────────────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def block_in_out_ch(chnl_mult, base_ch, block_idx):
    """Helper to compute input and output channels for a UNet block."""
    return base_ch * chnl_mult[block_idx], base_ch * chnl_mult[block_idx + 1]


class ResidualBlock(nn.Module):
    """Residual block with AdaIN conditioning."""
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, out_ch * 2)  # for AdaIN: scale and shift
        )

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, cond_emb):
        """
        x: [B, C, H, W]
        time_emb: [B, time_emb_dim]
        cond_emb: [B, cond_emb_dim] (global graph embedding)
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        # AdaIN conditioning from graph embedding
        cond_params = self.cond_mlp(cond_emb)
        scale, shift = cond_params.chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (scale[:, :, None, None] + 1) + shift[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        return F.interpolate(self.conv(x), scale_factor=2, mode='nearest')


# ── Graph Encoder ───────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """Encodes the constraint graph into a global embedding."""
    def __init__(self, in_dim, hidden_dim, num_layers=3, edge_dim=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.gnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.proj_in = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()

        for i in range(num_layers):
            self.gnns.append(
                GATConv(hidden_dim, hidden_dim, heads=2, concat=False,
                        edge_dim=edge_dim, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Final projection to get global embedding (we'll use mean pooling)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr=None):
        """
        x: [N, in_dim] node features
        edge_index: [2, E]
        edge_attr: [E, edge_dim] or None
        """
        h = self.proj_in(x)
        for i in range(self.num_layers):
            h = self.gnns[i](h, edge_index, edge_attr=edge_attr)
            h = self.norms[i](h)
            h = F.relu(h) + h  # residual connection
        # Global pooling: mean over nodes
        global_emb = h.mean(dim=0, keepdim=True)  # [1, hidden_dim]
        return global_emb, h  # return both global and node embeddings if needed


# ── UNet with Graph Conditioning ────────────────────────────────────────

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128,
                 ch_mults=(1, 2, 4, 8), num_res_blocks=2,
                 time_emb_dim=256, cond_emb_dim=None, dropout=0.1):
        """
        in_channels: input image channels (RGB=3)
        out_channels: output channels (same as in_channels for noise prediction)
        base_channels: base number of channels
        ch_mults: channel multipliers for each resolution
        num_res_blocks: number of residual blocks per resolution
        time_emb_dim: dimension of timestep embedding
        cond_emb_dim: dimension of conditioning embedding (graph embedding)
        dropout: dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.ch_mults = ch_mults
        self.num_res_blocks = num_res_blocks
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim if cond_emb_dim is not None else base_channels

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Graph encoder (we'll pass it externally, but we need to know its output dim)
        # We'll assume the graph encoder outputs cond_emb_dim
        # The graph encoder will be defined outside and passed in, or we can instantiate here.
        # For simplicity, we'll assume the conditioning embedding is provided as input to forward.
        # However, we need to encode the graph inside the model.
        # Let's instantiate the graph encoder here.
        self.graph_encoder = GraphEncoder(
            in_dim=cfg.NODE_FEATURE_DIM,
            hidden_dim=self.cond_emb_dim,
            num_layers=3,
            edge_dim=cfg.NUM_EDGE_TYPES,
            dropout=dropout
        )

        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsample blocks
        self.downs = nn.ModuleList()
        ch = base_channels
        for i, mult in enumerate(ch_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(ch, out_ch, time_emb_dim, self.cond_emb_dim))
                ch = out_ch
            if i != len(ch_mults) - 1:
                self.downs.append(Downsample(ch))

        # Middle block
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim, self.cond_emb_dim)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim, self.cond_emb_dim)

        # Upsample blocks
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.ups.append(ResidualBlock(ch, out_ch, time_emb_dim, self.cond_emb_dim))
                ch = out_ch
            if i != 0:
                self.ups.append(Upsample(ch))

        # Final projection
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x, t, graph_data):
        """
        x: [B, 3, H, W] noisy image
        t: [B] timesteps
        graph_data: PyG Data object containing graph information (x, edge_index, edge_attr, etc.)
        """
        # Timestep embedding
        t_emb = self.time_mlp(t)

        # Encode the constraint graph to get global embedding
        # We need to handle batched graph data.
        # For simplicity, we assume graph_data is a single graph (batch size 1).
        # In practice, we need to handle batching.
        # We'll assume the graph_data is already batched by PyG's Batch.
        # We'll encode the graph to get a global embedding per graph in the batch.
        global_emb, _ = self.graph_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None
        )  # global_emb: [num_graphs, cond_emb_dim]
        # If batched, global_emb has shape [B, cond_emb_dim]; if not, we need to expand.
        if global_emb.dim() == 2:
            pass  # already [B, cond_emb_dim]
        else:
            global_emb = global_emb.unsqueeze(0)  # [1, cond_emb_dim]
            # If batch size > 1, we need to repeat. We'll assume batch size matches.
            # For now, we assume the graph_data is for a single image (batch size 1).
            # We'll expand to match the batch size of x.
            if global_emb.shape[0] == 1 and x.shape[0] > 1:
                global_emb = global_emb.repeat(x.shape[0], 1)

        # UNet
        x = self.init_conv(x)
        h = [x]  # skip connections

        # Downsample
        for i, layer in enumerate(self.downs):
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb, global_emb)
            else:
                x = layer(x)
            h.append(x)

        # Middle
        x = self.mid_block1(x, t_emb, global_emb)
        x = self.mid_block2(x, t_emb, global_emb)

        # Upsample
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                # No skip connection
                x = layer(x, t_emb, global_emb)
            else:
                x = layer(x)

        # Final projection
        return self.final_conv(x)


# ── Diffusion Helper Functions ──────────────────────────────────────────

def extract(a, t, x_shape):
    """Extract coefficients from a based on timestep t and reshape to match image shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, beta_start, beta_end):
    """Linear beta schedule as proposed in the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    """
    Gaussian diffusion model for training and sampling.
    """
    def __init__(self, model, image_size, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # Move to same device as model
        self.to(self.model.device if hasattr(self.model, 'device') else torch.device('cpu'))

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data (add noise)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, graph_data, noise=None):
        """Compute loss at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, graph_data)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t, graph_data, t_index):
        """Sample a single step from the model."""
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t, graph_data) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, graph_data):
        """Generate samples by iterating through timesteps."""
        device = next(self.model.parameters()).device
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, graph_data, i)
            # Optionally clip to [-1, 1]
            img = torch.clamp(img, -1.0, 1.0)
        return img

    @torch.no_grad()
    def sample(self, batch_size=1, graph_data=None):
        """Generate samples."""
        return self.p_sample_loop((batch_size, self.image_size, self.image_size, 3), graph_data)

    @torch.no_grad()
    def sample_with_guidance(self, batch_size=1, graph_data=None, guidance_scale=2.5):
        """Generate samples with classifier-free guidance."""
        # We need to implement classifier-free guidance in the model.
        # For now, we'll just call sample without guidance.
        return self.sample(batch_size, graph_data)