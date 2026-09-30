"""
config.py — All hyperparameters and paths in one place.

Change values here to experiment. No need to edit other files.
"""

import os

# Base directory (one level up from pipeline/)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ── Paths ──────────────────────────────────────────────────────────────
DATA_PKL = os.path.join(BASE_DIR, "ResPlan.pkl")
SPLIT_JSON = os.path.join(BASE_DIR, "split.json")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


# ── Room & Edge Taxonomy ───────────────────────────────────────────────
ROOM_TYPES = [
    "balcony", "bathroom", "bedroom", "front_door",
    "kitchen", "living", "stair", "storage",
]
EDGE_TYPES = ["via_door", "adjacency", "via_window", "direct"]

NUM_ROOM_TYPES = len(ROOM_TYPES)   # 8
NUM_EDGE_TYPES = len(EDGE_TYPES)   # 4


# ── Model ──────────────────────────────────────────────────────────────
NODE_FEATURE_DIM = NUM_ROOM_TYPES + 1   # 8 one-hot type + 1 normalized area = 9
GNN_HIDDEN_DIM = 128                    # was 64 – more capacity for graph structure
LATENT_DIM = 32
NUM_GNN_LAYERS = 3
DROPOUT = 0.1                           # regularisation for GNN + MLP layers


# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 5e-4                    # was 1e-3 – gentler start
EPOCHS = 500
DEVICE = "cuda"                         # "cuda" or "cpu"

# KL annealing — ramp weight from 0 → KL_WEIGHT_MAX over training
KL_WEIGHT_MAX       = 0.05             # was fixed 0.001 – target KL weight
KL_ANNEAL_STRATEGY  = "cyclical"       # "monotonic" or "cyclical"
KL_ANNEAL_EPOCHS    = 100              # ramp length (monotonic) / cycle base (cyclical)
KL_ANNEAL_CYCLES    = 4                # number of cycles (cyclical only)

# LR schedule
LR_WARMUP_EPOCHS    = 10               # linear warmup before cosine decay
LR_MIN              = 1e-6             # minimum LR for cosine anneal

# Gradient clipping
GRAD_CLIP_NORM      = 1.0              # max gradient L2 norm

# Overlap loss weight
OVERLAP_WEIGHT = 0.1                   # penalty for room overlaps (start small)


# ── Evaluation ─────────────────────────────────────────────────────────
EVAL_EVERY = 5                          # was 10 – validate more frequently
SAVE_EVERY = 50                         # save a periodic checkpoint every N epochs
IOU_THRESHOLDS = [0.25, 0.50, 0.75]    # thresholds for precision / recall / F1
