"""Continuous Gaussian diffusion over room coordinates (plan §12, §17-20).

The network predicts the clean coordinates x0.  Training loss = masked x0 MSE
(primary, §17) + vertex-count CE + differentiable auxiliary geometry losses
(§18) evaluated on the x0 estimate and weighted by the signal level of the
sampled timestep (so only reasonably-denoised estimates are penalised).
"""
from __future__ import annotations
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MIN_VERTICES, ROOM_TYPE_TO_ID, ModelConfig, TrainConfig
from .data import area_features
from .losses import aux_losses, rect_to_polygon
from .model import FloorplanDenoiser

COORD_SCALE = 2.0          # x0 (in units of S) is multiplied by this -> roughly unit variance


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class FloorplanDiffusion(nn.Module):
    def __init__(self, cfg: ModelConfig, S: float):
        super().__init__()
        self.cfg, self.S = cfg, float(S)
        self.net = FloorplanDenoiser(cfg)
        self.register_buffer("alpha_bar", cosine_alphas_cumprod(cfg.timesteps), persistent=False)

    # -- targets --------------------------------------------------------------
    def targets(self, batch):
        """x0 [B,N,K,D] in network units (centred / S * COORD_SCALE) and token mask [B,N,K]."""
        if self.cfg.mode == "rect":
            x0 = batch["rect"][:, :, None, :]
            tok = batch["room_mask"][:, :, None]
        else:
            x0 = batch["coords"]
            tok = batch["vmask"]
        return x0 * COORD_SCALE, tok

    def to_polygons(self, x, tok_mask):
        """Network output [B,N,K,D] -> canvas-unit polygons P [B,N,V',2] and vertex mask."""
        x = x / COORD_SCALE
        if self.cfg.mode == "rect":
            P = rect_to_polygon(x[:, :, 0]) * self.S
            return P, torch.ones(P.shape[:3], dtype=torch.bool, device=P.device) & tok_mask.any(-1, keepdim=True)
        return x * self.S, tok_mask

    def encode(self, batch):
        af = area_features(batch["area"], self.S)
        return self.net.encode(batch["room_type"], af, batch["adj"], batch["room_mask"])

    # -- training -------------------------------------------------------------
    def training_losses(self, batch, tc: TrainConfig, aux_scale: float = 0.0, aux_samples: int = 16) -> Dict[str, torch.Tensor]:
        x0, tok = self.targets(batch)
        rmask = batch["room_mask"]
        B = x0.shape[0]
        room_h, glob_h, count_logits = self.encode(batch)
        t = torch.randint(0, self.cfg.timesteps, (B,), device=x0.device)
        ab = self.alpha_bar[t].view(B, 1, 1, 1)
        noise = torch.randn_like(x0)
        x_t = (ab.sqrt() * x0 + (1 - ab).sqrt() * noise) * tok.unsqueeze(-1)
        x0_hat = self.net.denoise(x_t, t, room_h, glob_h, rmask, tok)

        w = (tok & rmask[:, :, None]).unsqueeze(-1).float()
        l_diff = (((x0_hat.float() - x0) ** 2) * w).sum() / (w.sum() * x0.shape[-1]).clamp(min=1.0)
        out = {"diff": l_diff}
        total = l_diff

        if self.cfg.mode == "poly":
            ce = F.cross_entropy(count_logits.float()[rmask], batch["nverts"][rmask])
            out["count"] = ce
            total = total + tc.w_count * ce

        if aux_scale > 0:
            # use only the least-noisy samples of the batch (aux terms are costly and need a sane x0)
            k = min(aux_samples, B)
            idx = torch.argsort(t)[:k]
            P, vm = self.to_polygons(x0_hat[idx].float(), tok[idx])
            vm = vm & rmask[idx].unsqueeze(-1)
            exempt = batch["room_type"][idx] == ROOM_TYPE_TO_ID["front_door"]
            terms = aux_losses(P, vm, rmask[idx], batch["area"][idx], batch["adj"][idx], exempt=exempt)
            sw = self.alpha_bar[t[idx]]                           # signal weight in [0,1]
            weights = dict(area=tc.w_area, valid=tc.w_valid, overlap=tc.w_overlap,
                           conn=tc.w_conn, nonconn=tc.w_nonconn,
                           ortho=getattr(tc, 'w_ortho', 0.5), gap=getattr(tc, 'w_gap', 0.5))
            aux = 0.0
            for name, val in terms.items():
                v = (val * sw).mean()
                out["aux_" + name] = v
                aux = aux + weights[name] * v
            total = total + aux_scale * aux
        out["loss"] = total
        return out

    # -- sampling -------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch, steps: int = 50, eta: float = 0.0, sample_counts: bool = False,
               generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """DDIM (eta=0) / DDPM-like (eta=1) sampling of coordinates for the graphs in ``batch``.

        ``batch`` supplies only conditioning (types, areas, edges); any geometry in it is ignored.
        Returns normalised coordinates (units of S, centred) and the predicted vertex mask.
        """
        dev = batch["room_type"].device
        rmask = batch["room_mask"]
        B, N = rmask.shape
        K, D = self.cfg.tokens_per_room, self.cfg.coord_dim
        room_h, glob_h, count_logits = self.encode(batch)

        if self.cfg.mode == "rect":
            counts = torch.full((B, N), 4, device=dev, dtype=torch.long)
            tok = rmask[:, :, None].clone()
        else:
            logits = count_logits.float().clone()
            logits[..., :MIN_VERTICES] = float("-inf")
            if sample_counts:
                probs = logits.softmax(-1).reshape(-1, logits.shape[-1])
                counts = torch.multinomial(probs, 1, generator=generator).view(B, N)
            else:
                counts = logits.argmax(-1)
            tok = (torch.arange(K, device=dev)[None, None] < counts.unsqueeze(-1)) & rmask[:, :, None]

        x = torch.randn(B, N, K, D, device=dev, generator=generator) * tok.unsqueeze(-1)
        T = self.cfg.timesteps
        ts = torch.linspace(T - 1, 0, steps, device=dev).round().long()
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else None
            tb = t.expand(B)
            x0_hat = self.net.denoise(x, tb, room_h, glob_h, rmask, tok).float()
            x0_hat = x0_hat.clamp(-6.0, 6.0)
            ab_t = self.alpha_bar[t]
            if t_prev is None:
                x = x0_hat
                break
            ab_p = self.alpha_bar[t_prev]
            eps = (x - ab_t.sqrt() * x0_hat) / (1 - ab_t).sqrt().clamp(min=1e-4)
            sigma = eta * ((1 - ab_p) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_p).clamp(min=0).sqrt()
            dir_xt = (1 - ab_p - sigma ** 2).clamp(min=0).sqrt() * eps
            z = torch.randn(x.shape, device=dev, generator=generator)
            x = (ab_p.sqrt() * x0_hat + dir_xt + sigma * z) * tok.unsqueeze(-1)
        x = x * tok.unsqueeze(-1)
        return dict(x=x, tok_mask=tok, counts=counts)


def load_checkpoint(path: str, device: str = "cpu", use_ema: bool = True) -> "FloorplanDiffusion":
    """Rebuild a trained model (EMA weights by default) from a ``train.py`` checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FloorplanDiffusion(ModelConfig(**ck["model_cfg"]), ck["S"]).to(device)
    model.load_state_dict(ck["ema"] if use_ema and ck.get("ema") else ck["model"], strict=False)
    return model.eval()
