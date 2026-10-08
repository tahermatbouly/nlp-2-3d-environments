#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

echo "========================================"
echo "Starting Floorplan Diffusion Pipeline..."
echo "========================================"

# Phase 1: Train the base rectangle model
echo "--- [1/4] Running Phase 1 (Rectangles) ---"
python -m floorplan_diffusion.train \
    --phase 1 \
    --out_dir checkpoints/phase1 \
    --epochs 100

# Phase 2: Train the base polygon model
echo "--- [2/4] Running Phase 2 (Polygons) ---"
python -m floorplan_diffusion.train \
    --phase 2 \
    --out_dir checkpoints/phase2 \
    --epochs 200

# Phase 3: Fine-tune polygons with geometric auxiliary losses
# FIX 1: We remove the 40-epoch warmup since we initialize from a fully trained model,
# forcing the model to adapt to the physical constraints immediately.
echo "--- [3/4] Running Phase 3 (Auxiliary Losses) ---"
python -m floorplan_diffusion.train \
    --phase 3 \
    --init_from checkpoints/phase2/best.pt \
    --out_dir checkpoints/phase3 \
    --epochs 200 \
    --aux_start_epoch 0 \
    --aux_ramp_epochs 0

# Phase 4 & 5: Evaluate the final model on the test set
# FIX 2: We apply a strong geometric guidance scale during inference to actively 
# push overlapping rooms apart using gradient optimization at each diffusion step.
echo "--- [4/4] Running Evaluation ---"
python -m floorplan_diffusion.evaluate \
    --ckpt checkpoints/phase3/best.pt \
    --split test \
    --k 16 \
    --guidance 5.0 \
    --steps 50

echo "Pipeline complete!"
