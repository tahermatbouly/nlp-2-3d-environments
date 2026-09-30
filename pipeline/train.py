"""
train.py — Training loop for the Floor Plan CVAE.

Features
--------
  - Cyclical / monotonic KL annealing to prevent posterior collapse
  - Cosine LR schedule with linear warmup
  - Gradient clipping
  - Verbose per-epoch metrics: IoU, Precision, Recall, F1, overlap rate
  - Best model selection based on *generation* MSE (not reconstruction loss)
  - Overlap penalty to prevent room overlaps in generated floor plans

Usage
-----
    python train.py                  # full training (500 epochs)
    python train.py --epochs 5       # quick smoke test
"""

import os
import argparse
import time
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

import config as cfg
from dataset import FloorPlanDataset
from model import FloorPlanCVAE


# ── Overlap loss ────────────────────────────────────────────────────────

def box_overlap_loss(pred_boxes, eps=1e-6):
    """
    Compute differentiable overlap penalty for bounding boxes.
    pred_boxes: tensor of shape (N, 4) with format (cx, cy, w, h) normalized to [0,1]
    Returns: scalar loss encouraging non-overlap
    """
    if len(pred_boxes) < 2:
        return torch.tensor(0.0, device=pred_boxes.device)

    # Convert to corner coordinates: (x1, y1, x2, y2)
    x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2  # cx - w/2
    y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2  # cy - h/2
    x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2  # cx + w/2
    y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2  # cy + h/2

    # Compute pairwise overlaps
    overlap_loss = 0.0
    pair_count = 0

    for i in range(len(pred_boxes)):
        for j in range(i + 1, len(pred_boxes)):
            # Intersection dimensions
            ix1 = torch.max(x1[i], x1[j])
            iy1 = torch.max(y1[i], y1[j])
            ix2 = torch.min(x2[i], x2[j])
            iy2 = torch.min(y2[i], y2[j])

            # Intersection area (clamped to avoid negatives)
            iw = torch.clamp(ix2 - ix1, min=0.0)
            ih = torch.clamp(iy2 - iy1, min=0.0)
            intersection = iw * ih

            # Union area
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_j = (x2[j] - x1[j]) * (y2[j] - y1[j])
            union = area_i + area_j - intersection + eps  # eps for numerical stability

            # IoU (Intersection over Union)
            iou = intersection / union

            # Add to loss (we want to minimize overlap, so penalize high IoU)
            overlap_loss += iou
            pair_count += 1

    return overlap_loss / max(pair_count, 1)


# ── KL annealing ────────────────────────────────────────────────────────

def get_kl_weight(epoch: int, total_epochs: int) -> float:
    """Compute KL weight for the current epoch.

    Supports monotonic linear ramp and cyclical annealing
    (Fu et al., 2019 — "Cyclical Annealing Schedule").
    """
    if cfg.KL_ANNEAL_STRATEGY == "monotonic":
        return min(cfg.KL_WEIGHT_MAX,
                   cfg.KL_WEIGHT_MAX * epoch / max(cfg.KL_ANNEAL_EPOCHS, 1))
    else:
        # cyclical: repeat the ramp KL_ANNEAL_CYCLES times
        cycle_len = total_epochs / max(cfg.KL_ANNEAL_CYCLES, 1)
        tau = (epoch % cycle_len) / cycle_len
        # ramp during first half of each cycle, hold at max for second half
        return cfg.KL_WEIGHT_MAX * min(1.0, tau / 0.5)


# ── loss function ───────────────────────────────────────────────────────

def vae_loss(pred, target, mu, logvar, kl_weight, overlap_weight=0.0):
    """Combined reconstruction (MSE) + KL-divergence + overlap loss.

    Returns (total, recon, kl, overlap) so we can log each part separately.
    """
    recon = F.mse_loss(pred, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    overlap = box_overlap_loss(pred)
    total = recon + kl_weight * kl + overlap_weight * overlap
    return total, recon, kl, overlap


# ── IoU and metric helpers ──────────────────────────────────────────────

def compute_iou(b1, b2):
    """IoU between two (cx, cy, w, h) boxes (numpy arrays)."""
    x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x1b, y1b = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x2a, y2a = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2

    ix = max(0.0, min(x1b, x2b) - max(x1a, x2a))
    iy = max(0.0, min(y1b, y2b) - max(y1a, y2a))
    inter = ix * iy
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def compute_overlap_rate(bboxes):
    """Fraction of room pairs in a single plan that overlap (IoU > 0.01)."""
    n = len(bboxes)
    if n < 2:
        return 0.0
    pairs = 0
    overlapping = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if compute_iou(bboxes[i], bboxes[j]) > 0.01:
                overlapping += 1
    return overlapping / pairs if pairs > 0 else 0.0


def precision_recall_f1(ious, threshold):
    """Compute precision, recall, F1 at a given IoU threshold.

    For bounding-box regression with matched pairs (room i ↔ room i),
    a "true positive" means IoU ≥ threshold.
    """
    n = len(ious)
    if n == 0:
        return 0.0, 0.0, 0.0
    tp = sum(1 for iou in ious if iou >= threshold)
    precision = tp / n
    recall = tp / n      # same since rooms are 1-to-1 matched
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def room_type_from_onehot(feats):
    """Decode one-hot features back to room type string."""
    idx = int(feats[:cfg.NUM_ROOM_TYPES].argmax())
    return cfg.ROOM_TYPES[idx]


# ── validation with generation metrics ──────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, kl_weight, device, compute_gen_metrics=True):
    """Run validation and optionally compute generation-quality metrics.

    Returns a dict with all metrics.
    """
    model.eval()

    # Reconstruction metrics (via posterior z)
    v_loss, v_recon, v_kl, v_overlap, v_n = 0.0, 0.0, 0.0, 0.0, 0
    # mu/logvar stats
    all_mu, all_logvar = [], []

    for batch in val_loader:
        batch = batch.to(device)
        pred, mu, logvar = model(batch)
        loss, recon, kl, overlap = vae_loss(pred, batch.y, mu, logvar, kl_weight, cfg.OVERLAP_WEIGHT)
        v_loss  += loss.item()
        v_recon += recon.item()
        v_kl    += kl.item()
        v_overlap += overlap.item()
        v_n     += 1
        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())

    metrics = {
        "val_loss":  v_loss  / max(v_n, 1),
        "val_recon": v_recon / max(v_n, 1),
        "val_kl":    v_kl    / max(v_n, 1),
        "val_overlap": v_overlap / max(v_n, 1),
    }

    # mu / sigma statistics
    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_logvar = torch.cat(all_logvar, dim=0).numpy()
    all_sigma = np.exp(0.5 * all_logvar)
    metrics["mu_norm"]    = float(np.mean(np.abs(all_mu)))
    metrics["sigma_mean"] = float(np.mean(all_sigma))

    if not compute_gen_metrics:
        return metrics

    # ── generation metrics (z ~ N(0,I)) ────────────────────────────────
    gen_mse_sum = 0.0
    gen_n_plans = 0
    all_ious = []
    room_type_ious = {rt: [] for rt in cfg.ROOM_TYPES}
    overlap_rates = []

    for batch in val_loader:
        batch = batch.to(device)
        pred_gen = model.generate(batch)
        real = batch.y

        gen_mse_sum += F.mse_loss(pred_gen, real).item()
        gen_n_plans += 1

        pred_np = pred_gen.cpu().numpy()
        real_np = real.cpu().numpy()
        feats_np = batch.x.cpu().numpy()

        for p, r, f in zip(pred_np, real_np, feats_np):
            iou = compute_iou(p, r)
            all_ious.append(iou)
            rtype = room_type_from_onehot(f)
            room_type_ious[rtype].append(iou)

        # overlap rate per plan (for single-plan batches, this is exact;
        # for multi-plan batches, it's approximate over the full batch)
        overlap_rates.append(compute_overlap_rate(pred_np))

    metrics["gen_mse"] = gen_mse_sum / max(gen_n_plans, 1)
    metrics["gen_iou_mean"] = float(np.mean(all_ious)) if all_ious else 0.0
    metrics["gen_iou_median"] = float(np.median(all_ious)) if all_ious else 0.0
    metrics["overlap_rate"] = float(np.mean(overlap_rates)) if overlap_rates else 0.0

    # Precision / Recall / F1 at each IoU threshold
    for t in cfg.IOU_THRESHOLDS:
        p, r, f1 = precision_recall_f1(all_ious, t)
        key = f"{t:.2f}"
        metrics[f"prec@{key}"]   = p
        metrics[f"recall@{key}"] = r
        metrics[f"f1@{key}"]     = f1

    # Per-room-type IoU breakdown
    for rt, ious in room_type_ious.items():
        if ious:
            metrics[f"iou_{rt}"] = float(np.mean(ious))

    return metrics


# ── pretty logging ──────────────────────────────────────────────────────

def log_validation(epoch, total_epochs, train_metrics, val_metrics, kl_weight, lr, saved):
    """Print a structured, readable validation summary."""
    print(f"\nEpoch {epoch:4d}/{total_epochs}  lr={lr:.2e}  kl_w={kl_weight:.4f}")
    print(f"  ├─ Train  loss={train_metrics['loss']:.6f}  "
          f"recon={train_metrics['recon']:.6f}  kl={train_metrics['kl']:.4f}  "
          f"overlap={train_metrics.get('overlap', 0):.6f}")
    print(f"  ├─ Val    loss={val_metrics['val_loss']:.6f}  "
          f"recon={val_metrics['val_recon']:.6f}  kl={val_metrics['val_kl']:.4f}  "
          f"overlap={val_metrics.get('val_overlap', 0):.6f}")

    if "gen_mse" in val_metrics:
        print(f"  ├─ Gen    MSE={val_metrics['gen_mse']:.6f}  "
              f"IoU mean={val_metrics['gen_iou_mean']:.4f}  "
              f"median={val_metrics['gen_iou_median']:.4f}")

        iou_parts = []
        for t in cfg.IOU_THRESHOLDS:
            key = f"{t:.2f}"
            iou_parts.append(f"@{key}={val_metrics.get(f'f1@{key}', 0):.3f}")
        print(f"  ├─ F1:    {' '.join(iou_parts)}")

        prec_parts = []
        for t in cfg.IOU_THRESHOLDS:
            key = f"{t:.2f}"
            prec_parts.append(f"@{key}={val_metrics.get(f'prec@{key}', 0):.3f}")
        print(f"  ├─ Prec:  {' '.join(prec_parts)}")

        rec_parts = []
        for t in cfg.IOU_THRESHOLDS:
            key = f"{t:.2f}"
            rec_parts.append(f"@{key}={val_metrics.get(f'recall@{key}', 0):.3f}")
        print(f"  ├─ Rec:   {' '.join(rec_parts)}")

        print(f"  ├─ Overlap Rate: {val_metrics.get('overlap_rate', 0) * 100:.1f}%")

        # Per-room-type IoU
        rt_parts = []
        for rt in cfg.ROOM_TYPES:
            key = f"iou_{rt}"
            if key in val_metrics:
                rt_parts.append(f"{rt}={val_metrics[key]:.3f}")
        if rt_parts:
            print(f"  ├─ IoU by type: {' '.join(rt_parts)}")

    print(f"  └─ μ-norm={val_metrics.get('mu_norm', 0):.3f}  "
          f"σ-mean={val_metrics.get('sigma_mean', 0):.3f}"
          f"{'  ★ saved' if saved else ''}")


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Floor Plan CVAE")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                        help=f"Number of training epochs (default {cfg.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── data ────────────────────────────────────────────────────────────
    print("Loading dataset (this reads the 300 MB .pkl once)...")
    import pickle
    with open(cfg.DATA_PKL, "rb") as f:
        all_plans = pickle.load(f)

    train_ds = FloorPlanDataset(split="train", plans=all_plans)
    val_ds   = FloorPlanDataset(split="val",   plans=all_plans)
    del all_plans          # free ~300 MB

    print(f"Train : {len(train_ds)} plans")
    print(f"Val   : {len(val_ds)} plans")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── model ───────────────────────────────────────────────────────────
    model = FloorPlanCVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR scheduler: linear warmup → cosine decay
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=cfg.LR_WARMUP_EPOCHS,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - cfg.LR_WARMUP_EPOCHS, 1),
        eta_min=cfg.LR_MIN,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.LR_WARMUP_EPOCHS],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {n_params:,} trainable parameters")

    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    best_gen_mse = float("inf")

    # ── print config summary ────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  GNN hidden dim  : {cfg.GNN_HIDDEN_DIM}")
    print(f"  Latent dim      : {cfg.LATENT_DIM}")
    print(f"  Dropout         : {cfg.DROPOUT}")
    print(f"  LR              : {args.lr}  (warmup {cfg.LR_WARMUP_EPOCHS} → cosine → {cfg.LR_MIN})")
    print(f"  KL annealing    : {cfg.KL_ANNEAL_STRATEGY}  "
          f"max={cfg.KL_WEIGHT_MAX}  "
          f"{'cycles=' + str(cfg.KL_ANNEAL_CYCLES) if cfg.KL_ANNEAL_STRATEGY == 'cyclical' else 'ramp=' + str(cfg.KL_ANNEAL_EPOCHS)}")
    print(f"  Grad clip norm  : {cfg.GRAD_CLIP_NORM}")
    print(f"  Batch size      : {args.batch_size}")
    print(f"  Eval every      : {cfg.EVAL_EVERY} epochs")
    print(f"  IoU thresholds  : {cfg.IOU_THRESHOLDS}")
    print(f"{'═' * 60}\n")

    # ── training loop ───────────────────────────────────────────────────
    print(f"Starting training for {args.epochs} epochs …\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        kl_weight = get_kl_weight(epoch, args.epochs)

        # ── train ───────────────────────────────────────────────────────
        model.train()
        sum_loss, sum_recon, sum_kl, sum_overlap, n = 0.0, 0.0, 0.0, 0.0, 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred, mu, logvar = model(batch)
            loss, recon, kl, overlap = vae_loss(pred, batch.y, mu, logvar, kl_weight, cfg.OVERLAP_WEIGHT)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

            sum_loss  += loss.item()
            sum_recon += recon.item()
            sum_kl    += kl.item()
            sum_overlap += overlap.item()
            n += 1

        scheduler.step()

        train_m = {
            "loss":  sum_loss  / n,
            "recon": sum_recon / n,
            "kl":    sum_kl    / n,
            "overlap": sum_overlap / n,
        }

        # ── validate ────────────────────────────────────────────────────
        do_val = (epoch % cfg.EVAL_EVERY == 0) or (epoch == 1) or (epoch == args.epochs)

        if do_val:
            val_m = validate(model, val_loader, kl_weight, device,
                             compute_gen_metrics=True)

            saved = False
            gen_mse = val_m.get("gen_mse", float("inf"))
            if gen_mse < best_gen_mse:
                best_gen_mse = gen_mse
                torch.save(model.state_dict(),
                           os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt"))
                saved = True

            current_lr = optimizer.param_groups[0]["lr"]
            log_validation(epoch, args.epochs, train_m, val_m,
                           kl_weight, current_lr, saved)
        else:
            # compact one-line log for non-validation epochs
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train {train_m['loss']:.6f}  "
                  f"(recon {train_m['recon']:.6f}  kl {train_m['kl']:.4f}  "
                  f"overlap {train_m.get('overlap', 0):.6f})  "
                  f"lr={current_lr:.2e}  kl_w={kl_weight:.4f}")

        # periodic checkpoint
        if epoch % cfg.SAVE_EVERY == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_m["loss"],
                    "gen_mse": best_gen_mse,
                },
                os.path.join(cfg.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt"),
            )

    elapsed = time.time() - t0
    m, s = divmod(int(elapsed), 60)
    print(f"\n{'═' * 60}")
    print(f"Done in {m}m {s}s.  Best generation MSE: {best_gen_mse:.6f}")
    print(f"Best model → {os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pt')}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
