"""Training entry point.

    python -m floorplan_diffusion.train --phase 1                 # rectangles
    python -m floorplan_diffusion.train --phase 2                 # polygons
    python -m floorplan_diffusion.train --phase 3                 # polygons + geometric aux losses

Phase 4 (multi-sample ranking) and 5 (Shapely/NetworkX export) are inference-time:
see ``evaluate.py`` / ``generate.py``.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table

from .config import ModelConfig, TrainConfig
from .data import FloorplanDataset, collate, load_cache
from .diffusion import FloorplanDiffusion


def parse() -> argparse.Namespace:
    tc = TrainConfig()
    mc = ModelConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--mode", choices=["rect", "poly"], default=None, help="override phase default")
    for k, v in asdict(tc).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--d_model", type=int, default=mc.d_model)
    p.add_argument("--n_heads", type=int, default=mc.n_heads)
    p.add_argument("--graph_layers", type=int, default=mc.graph_layers)
    p.add_argument("--denoiser_layers", type=int, default=mc.denoiser_layers)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default="")
    p.add_argument("--quick_eval", type=int, default=32, help="#val plans for periodic generation check (0=off)")
    p.add_argument("--max_steps", type=int, default=0, help="stop after this many optimiser steps (smoke tests)")
    return p.parse_args()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                e.copy_(m)


def lr_at(step, tc: TrainConfig, total):
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    prog = (step - tc.warmup_steps) / max(1, total - tc.warmup_steps)
    return tc.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


@torch.no_grad()
def validate(model, loader, tc, device, aux_scale):
    model.eval()
    totals, n = {}, 0
    with torch.random.fork_rng(devices=[device] if device.startswith("cuda") else []):
        torch.manual_seed(1234)
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model.training_losses(b, tc, aux_scale=aux_scale)
            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def main():
    a = parse()
    tc = TrainConfig(**{k: getattr(a, k) for k in asdict(TrainConfig())})
    mode = a.mode or ("rect" if a.phase == 1 else "poly")
    mc = ModelConfig(mode=mode, d_model=a.d_model, n_heads=a.n_heads,
                     graph_layers=a.graph_layers, denoiser_layers=a.denoiser_layers)
    use_aux = a.phase >= 3
    torch.manual_seed(tc.seed); np.random.seed(tc.seed); random.seed(tc.seed)
    device = a.device

    cache = load_cache(tc.data, tc.split, tc.cache_dir)
    S = cache["S"]
    train = cache["train"][: tc.subset] if tc.subset else cache["train"]
    val = cache["val"][: max(64, tc.subset // 8)] if tc.subset else cache["val"]
    console = Console()
    console.print(f"[bold cyan][train][/bold cyan] mode={mode} phase={a.phase} train={len(train)} val={len(val)} S={S} device={device}")
    mk = lambda s, shuf: DataLoader(FloorplanDataset(s, S), batch_size=tc.batch_size, shuffle=shuf,
                                    num_workers=tc.num_workers, collate_fn=collate, drop_last=shuf,
                                    pin_memory=device.startswith("cuda"), persistent_workers=tc.num_workers > 0)
    tl, vl = mk(train, True), mk(val, False)

    model = FloorplanDiffusion(mc, S).to(device)
    console.print(f"[bold cyan][train][/bold cyan] parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.99))
    ema = EMA(model, tc.ema_decay)
    use_amp = tc.amp and device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    os.makedirs(tc.out_dir, exist_ok=True)
    start_epoch, step, best = 0, 0, float("inf")
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.shadow.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck.get("best", best)
        console.print(f"[bold cyan][train][/bold cyan] resumed from {a.resume} at epoch {start_epoch}")
    total_steps = tc.epochs * len(tl)
    log = open(os.path.join(tc.out_dir, "log.jsonl"), "a")

    def save(path, epoch):
        torch.save(dict(model=model.state_dict(), ema=ema.shadow.state_dict(), opt=opt.state_dict(),
                        epoch=epoch, step=step, best=best, model_cfg=mc.to_dict(), S=S,
                        train_cfg=asdict(tc), phase=a.phase), path)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        epoch_task = progress.add_task("[green]Epochs", total=tc.epochs, completed=start_epoch)
        batch_task = progress.add_task("[cyan]Batches", total=len(tl))

        for epoch in range(start_epoch, tc.epochs):
            progress.reset(batch_task)
            model.train()
            aux_scale = 0.0
            if use_aux and epoch >= tc.aux_start_epoch:
                aux_scale = min(1.0, (epoch - tc.aux_start_epoch + 1) / max(1, tc.aux_ramp_epochs))
            t0, run, cnt = time.time(), {}, 0
            for batch in tl:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, tc, total_steps)
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                    dtype=amp_dtype, enabled=use_amp):
                    out = model.training_losses(batch, tc, aux_scale=aux_scale)
                opt.zero_grad(set_to_none=True)
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(opt); scaler.update()
                ema.update(model)
                step += 1
                cnt += 1
                for k, v in out.items():
                    run[k] = run.get(k, 0.0) + float(v)
                progress.advance(batch_task)
                if a.max_steps and step >= a.max_steps:
                    break
            
            progress.advance(epoch_task)
            rec = {"epoch": epoch, "step": step, "time": round(time.time() - t0, 1), "aux_scale": aux_scale,
                   **{"train_" + k: v / max(cnt, 1) for k, v in run.items()}}
            
            if (epoch + 1) % tc.val_every == 0 or epoch == tc.epochs - 1 or (a.max_steps and step >= a.max_steps):
                v = validate(ema.shadow, vl, tc, device, aux_scale)
                rec.update({"val_" + k: x for k, x in v.items()})
                if v["diff"] < best:
                    best = v["diff"]; save(os.path.join(tc.out_dir, "best.pt"), epoch)
                if a.quick_eval:
                    from .metrics import evaluate_model
                    m = evaluate_model(ema.shadow, val[: a.quick_eval], K=2, steps=20, device=device)
                    rec["quick_csr_any_of_k"] = m["floorplan"]["csr_any_of_k"]
                    rec["quick_invalid_polygon_pct"] = m["floorplan"]["invalid_polygon_pct"]
                    rec["quick_connectivity_acc"] = m["floorplan"]["connectivity_acc"]
                
                table = Table(title=f"Epoch {epoch + 1} Summary", show_header=True, header_style="bold magenta", expand=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Train", justify="right")
                table.add_column("Val", justify="right")
                
                keys = [k.replace("train_", "") for k in rec.keys() if k.startswith("train_")]
                for k in sorted(keys):
                    t_val = f"{rec['train_' + k]:.4f}"
                    v_val = f"{rec.get('val_' + k, float('nan')):.4f}" if "val_"+k in rec else "-"
                    table.add_row(k, t_val, v_val)
                
                if a.quick_eval:
                    table.add_row("CSR (Any of K)", "-", f"{rec['quick_csr_any_of_k']:.3f}")
                    table.add_row("Invalid Poly %", "-", f"{rec['quick_invalid_polygon_pct']:.1f}%")
                
                progress.console.print(table)
                progress.update(epoch_task, description=f"[green]Epochs (loss: {rec.get('train_diff', 0.0):.4f})")
            else:
                progress.update(epoch_task, description=f"[green]Epochs (loss: {rec.get('train_diff', 0.0):.4f})")

            log.write(json.dumps(rec) + "\n"); log.flush()
            save(os.path.join(tc.out_dir, "last.pt"), epoch)
            if a.max_steps and step >= a.max_steps:
                break


if __name__ == "__main__":
    main()
