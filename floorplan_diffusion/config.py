"""Central configuration / constants for the floorplan diffusion project.

Coordinate-system note (DIFFUSION_PLAN.md §25):
    The ResPlan release stores every plan on a fixed 256-unit canvas (the
    metre conversion was deliberately dropped, see implementation plan).  So:
        dataset / Shapely / output : canvas units (``u``)
        neural network             : (x - plan_centroid) / S      (S fixed, dataset-wide)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict

# ---- vocabularies (defined once, kept consistent everywhere) ----------------
ROOM_TYPES = ["living", "bedroom", "bathroom", "kitchen", "balcony",
              "front_door", "storage", "stair"]
ROOM_TYPE_TO_ID: Dict[str, int] = {t: i for i, t in enumerate(ROOM_TYPES)}
NUM_ROOM_TYPES = len(ROOM_TYPES)

# Edge ids start at 1; 0 is reserved for "no edge" inside dense attention biases.
EDGE_TYPES = ["via_door", "adjacency", "direct", "via_window"]
EDGE_TYPE_TO_ID: Dict[str, int] = {t: i + 1 for i, t in enumerate(EDGE_TYPES)}
NUM_EDGE_TYPES = len(EDGE_TYPES)

# ---- size limits (measured on ResPlan: nodes max 22, vertices p99 = 26) ------
MAX_ROOMS = 24
MAX_VERTICES = 32
MIN_VERTICES = 3

# ---- geometry conventions ----------------------------------------------------
CANVAS = 256.0                 # plans live on a 256 x 256 canvas
COLLINEAR_TOL = 1e-3           # |cross| tolerance (units^2) for redundant-vertex removal
MIN_ROOM_AREA = 4.0            # rooms below this (units^2) are degenerate -> plan dropped

# Connectivity geometry, measured on ground truth (see implementation notes):
#   edge-connected rooms are separated by a wall gap of ~2.5-5 units;
#   `direct` rooms touch (distance 0); non-edges are never closer than 2 units.
CONNECT_TOL = {"via_door": 5.5, "adjacency": 5.0, "via_window": 6.0, "direct": 1.0}
NONEDGE_MARGIN = 2.0           # unconnected rooms should stay farther apart than this
SPURIOUS_TOL = 3.0             # shapely distance below which two unlinked rooms count as "touching"

# ---- validation / scoring (§22-23); weights are meant to be tuned -----------
# `front_door` is an entrance marker that legitimately overlaps `living` in the ground truth
# (199/200 plans), so it is exempt from overlap checks and the overlap loss.
OVERLAP_EXEMPT_TYPES = ("front_door",)
MAX_EXTENT_FACTOR = 1.25       # balconies may poke out of the 256 canvas (GT extent up to ~302)
AREA_REL_TOL = 0.30            # |A_pred - A_req| / A_req allowed
OVERLAP_REL_TOL = 0.10         # pair overlap area / smaller room area allowed
SCORE_WEIGHTS = dict(area=1.0, overlap=5.0, topology=2.0, invalid=10.0, compactness=0.5)


@dataclass
class ModelConfig:
    mode: str = "poly"               # "rect" (phase 1) or "poly" (phase 2+)
    d_model: int = 256
    n_heads: int = 8
    graph_layers: int = 4
    denoiser_layers: int = 6
    dropout: float = 0.0
    max_rooms: int = MAX_ROOMS
    max_vertices: int = MAX_VERTICES
    timesteps: int = 1000

    @property
    def tokens_per_room(self) -> int:
        return 1 if self.mode == "rect" else self.max_vertices

    @property
    def coord_dim(self) -> int:
        return 4 if self.mode == "rect" else 2

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    data: str = "ResPlan.pkl"
    split: str = "split.json"
    cache_dir: str = "cache"
    out_dir: str = "checkpoints/diffusion"
    batch_size: int = 32
    epochs: int = 200
    lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    aux_start_epoch: int = 20       # phase 3: auxiliary losses ramp in after this epoch
    aux_ramp_epochs: int = 20
    w_count: float = 0.1
    w_area: float = 1.0
    w_valid: float = 1.0
    w_overlap: float = 2.0
    w_conn: float = 1.0
    w_nonconn: float = 0.5
    w_ortho: float = 0.5
    w_gap: float = 0.5
    amp: bool = True
    num_workers: int = 4
    seed: int = 0
    val_every: int = 5
    subset: int = 0                 # >0: use only this many train plans (smoke tests)
    compile: bool = True
