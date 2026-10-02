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
# Node features: one-hot room type (8) + normalized area (1) = 9
NODE_FEATURE_DIM = NUM_ROOM_TYPES + 1   # 9
# GNN hidden dimension for encoding the constraint graph
GNN_HIDDEN_DIM = 256                    # Increased capacity for graph encoding
# Diffusion model channels (base width)
BASE_CHANNELS = 96
# Number of diffusion timesteps
TIMESTEPS = 250                         # Reduced for faster training (was 1000)
# Image size for generated floorplans
IMAGE_SIZE = 256
# Input channels to diffusion model (RGB = 3)
INPUT_CHANNELS = 3
# Whether to use classifier-free guidance
USE_CLASSIFIER_FREE_GUIDANCE = True
# Guidance strength for classifier-free guidance
GUIDANCE_STRENGTH = 2.5
# Dropout rate
DROPOUT = 0.1


# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE = 8                      # Reduced from 16 to avoid OOM (was 4, tried 16)
LEARNING_RATE = 2e-4                # Adjusted for diffusion training
LR_MIN = 1e-6                       # Minimum learning rate for cosine annealing
EPOCHS = 1000                       # More epochs may be needed
DEVICE = "cuda"                     # "cuda" or "cpu"

# Noise schedule (linear beta schedule)
BETA_START = 0.0001
BETA_END = 0.02

# Optimizer settings
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

# Gradient clipping
GRAD_CLIP_NORM = 1.0                # max gradient L2 norm

# EMA (Exponential Moving Average) for model weights
EMA_DECAY = 0.9999                  # Exponential moving average decay
EMA_START_STEP = 1000               # Start EMA after this many steps
EMA_UPDATE_EVERY = 10               # Update EMA every N steps


# ── Evaluation ─────────────────────────────────────────────────────────
EVAL_EVERY = 10                     # Validate every N epochs
SAVE_EVERY = 50                     # Save a periodic checkpoint every N epochs
NUM_SAMPLES_TO_GENERATE = 4         # How many samples to generate during evaluation


# ── Diffusion Loss Weights (if needed) ─────────────────────────────────
# We'll use simple L2 loss on the noise prediction, so no additional weights needed.
# If we want to weight different parts of the image, we can add here.
LOSS_TYPE = "l2"                    # Options: l2, l1, huber
# Gradient accumulation steps (set >1 to simulate larger batch size)
# GRADIENT_ACCUMULATION_STEPS = 4
