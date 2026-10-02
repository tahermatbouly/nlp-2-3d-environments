"""
train.py — Training loop for the Diffusion Floorplan Model.

Features
--------
- Diffusion model with graph conditioning (UNet + GAT encoder)
- Beautiful terminal output with loss, sampling progress, and eta
- Periodic sample generation and saving
- Checkpointing based on latest epoch (or best sample quality if metrics implemented)
- EMA (Exponential Moving Average) for model weights
- LR scheduler with warmup and cosine decay
"""

import os
import sys
import time
import math
from datetime import timedelta

# Add the project root to sys.path so we can import the pipeline package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Optional, but we can use simple logging

from pipeline import config as cfg
from pipeline.dataset import FloorPlanDataset
from pipeline.model import UNet, GaussianDiffusion, GraphEncoder

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# Optional: for EMA
class EMA:
    def __init__(self, model, decay=0.9999, start_step=1000, update_every=10):
        self.model = model
        self.decay = decay
        self.start_step = start_step
        self.update_every = update_every
        self.shadow = {}
        self.step = 0
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.step += 1
        if self.step < self.start_step:
            return
        if self.step % self.update_every != 0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.shadow[name]


def get_graph_embedding(batch, graph_encoder, device):
    """Compute graph embedding from batch data."""
    global_emb, _ = graph_encoder(
        batch.x.to(device),
        batch.edge_index.to(device),
        edge_attr=batch.edge_attr.to(device) if hasattr(batch, "edge_attr") else None,
        batch_index=batch.batch.to(device) if hasattr(batch, "batch") else None
    )
    return global_emb


def main():
    # ── Setup ─────────────────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load dataset ───────────────────────────────────────────────────────
    print("Loading dataset (this reads the 300 MB .pkl once)...")
    import pickle
    with open(cfg.DATA_PKL, "rb") as f:
        all_plans = pickle.load(f)

    train_ds = FloorPlanDataset(split="train", plans=all_plans)
    val_ds   = FloorPlanDataset(split="val",   plans=all_plans)
    del all_plans  # free memory

    print(f"Train : {len(train_ds)} plans")
    print(f"Val   : {len(val_ds)} plans")

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=min(8, os.cpu_count()), pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=min(4, os.cpu_count()), pin_memory=True)

    # ── Initialize model and diffusion ─────────────────────────────────────
    print("Initializing model...")
    # We need to determine the conditioning embedding dimension.
    # Let's set it to the base channels for simplicity.
    cond_emb_dim = cfg.BASE_CHANNELS
    denoise_model = UNet(
        in_channels=cfg.INPUT_CHANNELS,
        out_channels=cfg.INPUT_CHANNELS,
        base_channels=cfg.BASE_CHANNELS,
        ch_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=cfg.TIME_EMB_DIM if hasattr(cfg, 'TIME_EMB_DIM') else 256,
        cond_emb_dim=cond_emb_dim,
        dropout=cfg.DROPOUT
    ).to(device)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(torch, "compile"):
            denoise_model = torch.compile(denoise_model)

    # Create separate graph encoder for precomputing embeddings (to avoid redundant computation in model)
    graph_encoder = GraphEncoder(
        in_dim=cfg.NUM_ROOM_TYPES + 1,  # room type one-hot + normalized area
        hidden_dim=cond_emb_dim,
        num_layers=3,
        edge_dim=cfg.NUM_EDGE_TYPES,
        dropout=0.1
    ).to(device)

    diffusion = GaussianDiffusion(
        model=denoise_model,
        image_size=cfg.IMAGE_SIZE,
        timesteps=cfg.TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END
    ).to(device)

    # EMA
    ema = EMA(denoise_model, decay=cfg.EMA_DECAY, start_step=cfg.EMA_START_STEP, update_every=cfg.EMA_UPDATE_EVERY)

    # Optimizer and scheduler
    optimizer = Adam(list(denoise_model.parameters()) + list(graph_encoder.parameters()), lr=cfg.LEARNING_RATE, betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2))
    total_steps = len(train_loader) * cfg.EPOCHS
    warmup_steps = len(train_loader) * cfg.LR_WARMUP_EPOCHS if hasattr(cfg, 'LR_WARMUP_EPOCHS') else 0

    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, total_iters=max(warmup_steps, 1)),
                CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=cfg.LR_MIN)
            ],
            milestones=[max(warmup_steps, 1)]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.LR_MIN)

    # Logging setup
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    log_dir = os.path.join(cfg.CHECKPOINT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if 'SummaryLoader' in globals() else None  # Simple check

    # Training state
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    # ── Print config summary ───────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  Image size        : {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"  Timesteps         : {cfg.TIMESTEPS}")
    print(f"  Base channels     : {cfg.BASE_CHANNELS}")
    print(f"  LR                : {cfg.LEARNING_RATE}  (warmup {cfg.LR_WARMUP_EPOCHS if hasattr(cfg, 'LR_WARMUP_EPOCHS') else 0} → cosine → {cfg.LR_MIN})")
    print(f"  Batch size        : {cfg.BATCH_SIZE}")
    print(f"  EMA decay         : {cfg.EMA_DECAY}")
    print(f"  Grad clip norm    : {cfg.GRAD_CLIP_NORM}")
    print(f"{'═' * 70}\n")

    # ── Training loop ─────────────────────────────────────────────────────
    console = Console()
    console.print(f"[bold green]Starting training for {cfg.EPOCHS} epochs...[/bold green]")
    t0 = time.time()

    use_amp = getattr(cfg, 'USE_AMP', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    grad_accum_steps = getattr(cfg, 'GRADIENT_ACCUMULATION_STEPS', 1)

    epoch_progress = tqdm(range(start_epoch, cfg.EPOCHS + 1), desc="Epochs", colour="cyan")
    for epoch in epoch_progress:
        epoch_start_time = time.time()
        denoise_model.train()
        epoch_loss = 0.0
        num_batches = 0

        # ── Training ───────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = batch.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Precompute graph embedding for the entire batch (to avoid redundant computation)
                global_emb = get_graph_embedding(batch, graph_encoder, device)

                # Sample random timesteps for each image in the batch
                t = torch.randint(0, diffusion.timesteps, (batch.target_img.shape[0],), device=device).long()

                # Compute loss using precomputed graph embedding
                loss = diffusion.p_losses(
                    x_start=batch.target_img,
                    t=t,
                    graph_data=batch,  # Still needed for compatibility, but model will ignore if cond_emb provided
                    cond_emb=global_emb
                )
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            num_batches += 1
            global_step += 1

            if num_batches % grad_accum_steps == 0 or num_batches == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(denoise_model.parameters(), cfg.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update()

            epoch_loss += loss.item() * grad_accum_steps

            # Update progress bar
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                epoch_progress.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})

            # Optional: log to tensorboard
            if writer is not None and global_step % 100 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)

        avg_train_loss = epoch_loss / max(num_batches, 1)

        # ── Validation ─────────────────────────────────────────────────────
        denoise_model.eval()
        ema.apply_shadow()
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            val_loss = 0.0
            ema_val_loss = 0.0
            val_batches = 0
            for batch in val_loader:
                batch = batch.to(device)
                # Precompute graph embedding once per batch
                global_emb = get_graph_embedding(batch, graph_encoder, device)

                t = torch.randint(0, diffusion.timesteps, (batch.target_img.shape[0],), device=device).long()
                # Regular model loss
                loss = diffusion.p_losses(
                    x_start=batch.target_img,
                    t=t,
                    graph_data=batch,  # Still needed for compatibility, but model will ignore if cond_emb provided
                    cond_emb=global_emb
                )
                val_loss += loss.item()

                # EMA model loss
                ema_loss = diffusion.p_losses(
                    x_start=batch.target_img,
                    t=t,
                    graph_data=batch,  # Still needed for compatibility, but model will ignore if cond_emb provided
                    cond_emb=global_emb
                )
                ema_val_loss += ema_loss.item()
                val_batches += 1
            avg_val_loss = val_loss / max(val_batches, 1)
            ema_avg_val_loss = ema_val_loss / max(val_batches, 1)
        ema.restore()

        # ── Logging ────────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - t0
        eta_seconds = (total_time / max(1, epoch)) * (cfg.EPOCHS - epoch)
        eta = timedelta(seconds=int(eta_seconds))

        print(f"Epoch {epoch:4d}/{cfg.EPOCHS}  "
              f"Loss: {avg_train_loss:.6f}  "
              f"Val Loss: {avg_val_loss:.6f}  "
              f"EMA Val Loss: {ema_avg_val_loss:.6f}  "
              f"LR: {scheduler.get_last_lr()[0]:.2e}  "
              f"Time: {epoch_time:.1f}s  "
              f"ETA: {eta}")

        # Optional: log to tensorboard
        if writer is not None:
            writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Loss/ema_val", ema_avg_val_loss, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

            # Update epoch progress bar with metrics
            epoch_progress.set_postfix({
                'loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'ema_val_loss': f'{ema_avg_val_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        # ── Save samples periodically ───────────────────────────────────────
        if epoch % cfg.SAVE_EVERY == 0 or epoch == cfg.EPOCHS:
            print(f"  Generating samples at epoch {epoch}...")
            denoise_model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Use a few validation samples for conditioning
                val_samples = next(iter(val_loader))
                val_samples = val_samples.to(device)
                
                # Precompute graph embedding for validation samples
                global_emb = get_graph_embedding(val_samples, graph_encoder, device)
                # Generate samples
                sampled_images = diffusion.sample_with_guidance(
                    batch_size=val_samples.num_graphs,
                    graph_data=val_samples,
                    cond_emb=global_emb,
                    guidance_scale=cfg.GUIDANCE_STRENGTH if hasattr(cfg, "GUIDANCE_STRENGTH") else 2.5
                )
                # Denormalize from [-1,1] to [0,1]
                sampled_images = (sampled_images + 1.0) / 2.0
                sampled_images = torch.clamp(sampled_images, 0.0, 1.0)


                # Save as grid
                try:
                    import torchvision
                    grid = torchvision.utils.make_grid(sampled_images, nrow=2, pad_value=1.0)
                    grid_path = os.path.join(cfg.CHECKPOINT_DIR, f"samples_epoch_{epoch}.png")
                    torchvision.utils.save_image(grid, grid_path)
                    print(f"    Saved samples to {grid_path}")
                except ImportError:
                    # Fallback: save each image separately
                    for i, img in enumerate(sampled_images):
                        img_path = os.path.join(cfg.CHECKPOINT_DIR, f"sample_epoch_{epoch}_{i}.png")
                        # We'll need to convert tensor to PIL and save; skip for simplicity
                        pass

        # ── Save checkpoint ───────────────────────────────────────────────
        if epoch % cfg.SAVE_EVERY == 0 or epoch == cfg.EPOCHS:
            ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
            
            # Unwrap compiled model for clean state_dict
            model_to_save = denoise_model._orig_mod if hasattr(denoise_model, '_orig_mod') else denoise_model
            
            save_dict = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema.shadow if hasattr(ema, 'shadow') else {},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
            }
            
            torch.save(save_dict, ckpt_path)
            print(f"    Saved checkpoint to {ckpt_path}")

            # Also save as latest
            latest_path = os.path.join(cfg.CHECKPOINT_DIR, "latest.pt")
            torch.save(save_dict, latest_path)

    # ── Final ─────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    m, s = divmod(int(total_time), 60)
    print(f"\n{'═' * 70}")
    print(f"Done in {m}m {s}s.  Best loss: {best_loss:.6f}")
    print(f"Latest checkpoint → {os.path.join(cfg.CHECKPOINT_DIR, 'latest.pt')}")
    print(f"{'═' * 70}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()