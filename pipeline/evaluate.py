"""
evaluate.py — Comprehensive evaluation of the trained CVAE.

Reports
-------
  - Generation MSE (z ~ N(0,I)), Reconstruction MSE (z from posterior)
  - Per-room-type IoU breakdown
  - Precision / Recall / F1 at IoU thresholds [0.25, 0.50, 0.75]
  - Room overlap analysis
  - μ / σ posterior statistics
  - Side-by-side visual comparisons (ground truth vs generated)

Usage
-----
    python evaluate.py                # metrics on the test set + sample images
    python evaluate.py --n-samples 8  # show more side-by-side comparisons
"""

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.loader import DataLoader

import config as cfg
from dataset import FloorPlanDataset
from model import FloorPlanCVAE


# ── room colours (same palette as the original dataset) ────────────────

ROOM_COLORS = {
    "living":     "#d9d9d9",
    "bedroom":    "#66c2a5",
    "bathroom":   "#fc8d62",
    "kitchen":    "#8da0cb",
    "balcony":    "#b3b3b3",
    "front_door": "#a63603",
    "storage":    "#FF8C69",
    "stair":      "#9e9ac8",
}


# ── helpers ─────────────────────────────────────────────────────────────

def _room_type_name(onehot):
    """Decode a one-hot vector back to a room-type string."""
    idx = int(onehot[:cfg.NUM_ROOM_TYPES].argmax())
    return cfg.ROOM_TYPES[idx]


def draw_layout(bboxes, node_features, title="", ax=None):
    """Draw normalised bounding boxes as coloured rectangles."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    for bbox, feats in zip(bboxes, node_features):
        cx, cy, w, h = bbox
        rtype = _room_type_name(feats)
        color = ROOM_COLORS.get(rtype, "#cccccc")

        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.005",
            linewidth=1.5, edgecolor="black",
            facecolor=color, alpha=0.75,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, rtype, ha="center", va="center",
                fontsize=6, fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.invert_yaxis()
    return ax


def compute_iou(b1, b2):
    """IoU between two (cx, cy, w, h) boxes."""
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
    """Fraction of room pairs that overlap (IoU > 0.01)."""
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
    """Precision, Recall, F1 at a given IoU threshold (matched pairs)."""
    n = len(ious)
    if n == 0:
        return 0.0, 0.0, 0.0
    tp = sum(1 for iou in ious if iou >= threshold)
    precision = tp / n
    recall = tp / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of side-by-side comparisons to show")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val", "train"],
                        help="Which split to evaluate on (default: test)")
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    # ── load model ──────────────────────────────────────────────────────
    model_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at {model_path}")
        print("Run  python train.py  first.")
        return

    model = FloorPlanCVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── load data ───────────────────────────────────────────────────────
    print(f"Loading {args.split} split …")
    eval_ds = FloorPlanDataset(split=args.split)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)
    print(f"{args.split.title()} plans: {len(eval_ds)}")

    # ── quantitative metrics ────────────────────────────────────────────
    # Generation metrics (z ~ N(0,I))
    gen_mse_sum, gen_n = 0.0, 0
    all_gen_ious = []
    room_type_gen_ious = {rt: [] for rt in cfg.ROOM_TYPES}
    overlap_rates = []

    # Reconstruction metrics (z from posterior)
    recon_mse_sum, recon_n = 0.0, 0
    all_recon_ious = []
    room_type_recon_ious = {rt: [] for rt in cfg.ROOM_TYPES}
    all_mu, all_logvar = [], []

    with torch.no_grad():
        for data in eval_loader:
            data = data.to(device)
            real = data.y
            feats = data.x

            # ── generation path ─────────────────────────────────────────
            pred_gen = model.generate(data)
            gen_mse_sum += F.mse_loss(pred_gen, real).item()
            gen_n += 1

            pred_gen_np = pred_gen.cpu().numpy()
            real_np = real.cpu().numpy()
            feats_np = feats.cpu().numpy()

            for p, r, f in zip(pred_gen_np, real_np, feats_np):
                iou = compute_iou(p, r)
                all_gen_ious.append(iou)
                rtype = _room_type_name(f)
                room_type_gen_ious[rtype].append(iou)

            overlap_rates.append(compute_overlap_rate(pred_gen_np))

            # ── reconstruction path ─────────────────────────────────────
            pred_recon, mu, logvar = model(data)
            recon_mse_sum += F.mse_loss(pred_recon, real).item()
            recon_n += 1

            pred_recon_np = pred_recon.cpu().numpy()
            for p, r in zip(pred_recon_np, real_np):
                all_recon_ious.append(compute_iou(p, r))

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    # ── aggregate stats ─────────────────────────────────────────────────
    all_mu_cat = torch.cat(all_mu, dim=0).numpy()
    all_logvar_cat = torch.cat(all_logvar, dim=0).numpy()
    all_sigma = np.exp(0.5 * all_logvar_cat)

    # ── print results ───────────────────────────────────────────────────
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION RESULTS — {args.split.upper()} SET")
    print(f"  {gen_n} plans, {len(all_gen_ious)} rooms")
    print(f"{sep}")

    print(f"\n┌─ Reconstruction (z from posterior) ────────────────────────")
    print(f"│  MSE      : {recon_mse_sum / max(recon_n, 1):.6f}")
    print(f"│  IoU mean : {np.mean(all_recon_ious):.4f}")
    print(f"│  IoU med  : {np.median(all_recon_ious):.4f}")

    print(f"\n┌─ Generation (z ~ N(0,I)) ────────────────────────────────")
    print(f"│  MSE      : {gen_mse_sum / max(gen_n, 1):.6f}")
    print(f"│  IoU mean : {np.mean(all_gen_ious):.4f}")
    print(f"│  IoU med  : {np.median(all_gen_ious):.4f}")

    print(f"\n┌─ Precision / Recall / F1 (Generation) ────────────────────")
    for t in cfg.IOU_THRESHOLDS:
        p, r, f1 = precision_recall_f1(all_gen_ious, t)
        print(f"│  @{t:.2f}  Prec={p:.4f}  Rec={r:.4f}  F1={f1:.4f}")

    print(f"\n┌─ Room Overlap ────────────────────────────────────────────")
    print(f"│  Mean overlap rate: {np.mean(overlap_rates) * 100:.1f}%")
    print(f"│  Max  overlap rate: {np.max(overlap_rates) * 100:.1f}%")

    print(f"\n┌─ Per-Room-Type IoU (Generation) ──────────────────────────")
    for rt in cfg.ROOM_TYPES:
        ious = room_type_gen_ious[rt]
        if ious:
            print(f"│  {rt:12s}  mean={np.mean(ious):.4f}  "
                  f"med={np.median(ious):.4f}  n={len(ious)}")

    print(f"\n┌─ Posterior Statistics ─────────────────────────────────────")
    print(f"│  μ  mean={all_mu_cat.mean():.4f}  "
          f"|μ| mean={np.mean(np.abs(all_mu_cat)):.4f}  "
          f"std={all_mu_cat.std():.4f}")
    print(f"│  σ  mean={all_sigma.mean():.4f}  "
          f"std={all_sigma.std():.4f}  "
          f"min={all_sigma.min():.4f}  max={all_sigma.max():.4f}")
    print(f"│  (ideal: |μ|→0, σ→1)")
    print(f"└{'─' * 59}")

    # ── visual comparisons ──────────────────────────────────────────────
    n = min(args.n_samples, len(eval_ds))
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        data = eval_ds[i].to(device)
        feats = data.x.cpu().numpy()

        # real
        draw_layout(data.y.cpu().numpy(), feats,
                    title=f"Real  (plan {data.plan_id})", ax=axes[0, i])

        # generated
        with torch.no_grad():
            pred = model.generate(data)
        pred_np = pred.cpu().numpy()
        real_np = data.y.cpu().numpy()

        # compute IoU for this sample
        sample_ious = [compute_iou(p, r)
                       for p, r in zip(pred_np, real_np)]
        mean_iou = np.mean(sample_ious)
        draw_layout(pred_np, feats,
                    title=f"Generated  (plan {data.plan_id}, IoU={mean_iou:.3f})",
                    ax=axes[1, i])

    plt.tight_layout()
    out_path = os.path.join(cfg.CHECKPOINT_DIR, "sample_outputs.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visual comparison → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
