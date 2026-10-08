#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================"
echo "Starting Floorplan Diffusion Smoke Test..."
echo "========================================"

# Phase 1: Train the base rectangle model
echo "--- [1/4] Running Phase 1 (Rectangles) ---"
python -m floorplan_diffusion.train \
    --phase 1 \
    --out_dir checkpoints/smoke_phase1 \
    --max_steps 5

# Phase 2: Train the base polygon model
echo "--- [2/4] Running Phase 2 (Polygons) ---"
python -m floorplan_diffusion.train \
    --phase 2 \
    --out_dir checkpoints/smoke_phase2 \
    --max_steps 5

# Phase 3: Fine-tune polygons with geometric auxiliary losses
echo "--- [3/4] Running Phase 3 (Auxiliary Losses) ---"
python -m floorplan_diffusion.train \
    --phase 3 \
    --init_from checkpoints/smoke_phase2/best.pt \
    --out_dir checkpoints/smoke_phase3 \
    --max_steps 5 \
    --aux_start_epoch 0 \
    --aux_ramp_epochs 0

# Phase 4 & 5: Evaluate the final model on the test set
echo "--- [4/4] Running Evaluation ---"
python -m floorplan_diffusion.evaluate \
    --ckpt checkpoints/smoke_phase3/best.pt \
    --split test \
    --k 2 \
    --n 2 \
    --guidance 5.0 \
    --steps 5

echo "Smoke Test complete!"
