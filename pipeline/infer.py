"""
infer.py — Generate a floorplan image from a JSON description using a diffusion model.

Usage
-----
    # From an existing JSON file in training-data/
    python infer.py --json ../training-data/42.json

    # Generate multiple samples with guidance
    python infer.py --json ../training-data/42.json --num-samples 3 --guidance-scale 3.0

    # From a hand-written JSON (only needs "rooms" with types, areas, connections)
    python infer.py --json my_house.json

    # Save without displaying
    python infer.py --json ../training-data/42.json --output my_plan.png
"""

import os
import json
import argparse

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import config as cfg
from model import UNet, GaussianDiffusion
from dataset import FloorPlanDataset  # For json_to_data function? We'll rewrite it.
from torch_geometric.data import Data


# ── Recreate json_to_data from the old infer.py (adapted for our Data format) ────────
def json_to_data(desc: dict, device: torch.device) -> Data:
    """Turn a simplified JSON description into a PyG Data object for conditioning.

    The JSON only needs a ``rooms`` list.  Each room needs:
        id, type, area, connections[{target_id, type}]

    No ground-truth bounding boxes are required.
    """
    rooms = desc["rooms"]

    # map room string IDs → integer indices
    id_to_idx = {r["id"]: i for i, r in enumerate(rooms)}

    # ── node features ───────────────────────────────────────────────
    max_area = max(r["area"] for r in rooms) or 1.0
    node_features = []

    for r in rooms:
        rtype = r["type"]
        type_idx = (cfg.ROOM_TYPES.index(rtype)
                    if rtype in cfg.ROOM_TYPES else 0)
        onehot = [0.0] * cfg.NUM_ROOM_TYPES
        onehot[type_idx] = 1.0
        area_norm = r["area"] / max_area
        node_features.append(onehot + [area_norm])

    # ── edges ───────────────────────────────────────────────────────
    src, dst, edge_attrs = [], [], []
    seen = set()

    for r in rooms:
        u = id_to_idx[r["id"]]
        for conn in r.get("connections", []):
            tid = conn["target_id"]
            if tid not in id_to_idx:
                continue
            v = id_to_idx[tid]

            pair = (min(u, v), max(u, v))
            if pair in seen:
                continue
            seen.add(pair)

            etype = conn.get("type", "adjacency")
            if etype in cfg.EDGE_TYPES:
                etype_idx = cfg.EDGE_TYPES.index(etype)
            else:
                etype_idx = cfg.EDGE_TYPES.index("adjacency")

            onehot_e = [0.0] * cfg.NUM_EDGE_TYPES
            onehot_e[etype_idx] = 1.0

            # both directions (undirected)
            src.extend([u, v])
            dst.extend([v, u])
            edge_attrs.extend([onehot_e, onehot_e])

    if not edge_attrs:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, cfg.NUM_EDGE_TYPES), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        # We don't have target_image for inference, but the model expects it in the Data object?
        # We'll set it to zero tensor; the diffusion model doesn't use it during sampling.
        target_image=torch.zeros((3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=torch.float),
        plan_id=desc.get("id", "?"),
        plan_bounds=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float),  # dummy
    )
    return data.to(device)


# ── Helper to denormalize and save image ──────────────────────────────────────
def save_image(tensor, path):
    """Save a torch tensor [3, H, W] in [-1, 1] as a PNG image."""
    # Denormalize to [0, 1]
    img = (tensor + 1.0) / 2.0
    img = torch.clamp(img, 0.0, 1.0)
    # Convert to numpy and transpose to HWC
    img_np = img.permute(1, 2, 0).cpu().numpy()
    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    # Save using PIL
    img_pil = Image.fromarray(img_np)
    img_pil.save(path)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate a floorplan image from a JSON description.")
    parser.add_argument("--json", required=True,
                        help="Path to a JSON file describing the house")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of different layouts to generate "
                             "(each uses a different random seed)")
    parser.add_argument("--guidance-scale", type=float, default=2.5,
                        help="Classifier-free guidance scale (higher = stronger conditioning)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save image(s) to this path (if num-samples>1, appends index)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint "
                             "(default: checkpoints/latest.pt)")
    args = parser.parse_args()

    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model_path = args.model or os.path.join(cfg.CHECKPOINT_DIR, "latest.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}")
        print("Run  python train.py  first.")
        return

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model architecture (must match training)
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
    denoise_model.load_state_dict(checkpoint['model_state_dict'])
    denoise_model.eval()

    # Initialize diffusion
    diffusion = GaussianDiffusion(
        model=denoise_model,
        image_size=cfg.IMAGE_SIZE,
        timesteps=cfg.TIMESTEPS,
        beta_start=cfg.BETA_START,
        beta_end=cfg.BETA_END
    ).to(device)

    # ── Load JSON description ─────────────────────────────────────────────
    with open(args.json) as f:
        desc = json.load(f)

    room_types = [r["type"] for r in desc["rooms"]]
    plan_id = desc.get("id", "?")
    print(f"Plan {plan_id}: {len(desc['rooms'])} rooms — "
          f"{', '.join(room_types)}")

    # ── Convert to conditioning data ───────────────────────────────────────
    data = json_to_data(desc, device)

    # ── Generate layouts ───────────────────────────────────────────────────
    n = args.num_samples
    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))

    for i, ax in enumerate(axes):
        # Set seed for reproducibility if needed
        if args.num_samples > 1:
            torch.manual_seed(i + 42)  # different seed per sample

        # Sample with guidance
        with torch.no_grad():
            sample = diffusion.sample_with_guidance(
                batch_size=1,
                graph_data=data,
                guidance_scale=args.guidance_scale
            )  # shape [1, 3, H, W]
            sample = sample.squeeze(0)  # [3, H, W]

        # Save or display
        if args.output:
            if n == 1:
                save_path = args.output
            else:
                base, ext = os.path.splitext(args.output)
                save_path = f"{base}_{i}{ext}"
            save_image(sample, save_path)
            print(f"Saved → {save_path}")
        else:
            # Display using matplotlib
            ax.imshow(np.transpose(sample.cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)  # [0,1]
            ax.set_title(f"Generated layout (sample {i + 1})")
            ax.axis('off')

            # Print coordinates? We could extract bounding boxes from the image, but skip for now.
            # Instead, we can print a message.
            if i == 0:
                print(f"Generated {n} sample(s).")

    if not args.output:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()