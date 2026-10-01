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
# from model import FloorPlanCVAE  # Commented out - diffusion model uses different architecture
# For diffusion model evaluation, use sample generation in train.py or implement diffusion-specific evaluation


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
    print("Evaluation script for diffusion model is not yet implemented.")
    print("To evaluate the diffusion model, you can:")
    print("  1. Use the sample generation in train.py (it generates samples periodically)")
    print("  2. Run python pipeline/infer.py to generate samples from a checkpoint")
    print("  3. Implement diffusion-specific evaluation metrics (FID, precision/recall, etc.)")
    print("")
    print("For training, run: python pipeline/train.py")




if __name__ == "__main__":
    main()
