## Goal Description
The diffusion model currently generates realistic floorplans, but some rooms exhibit triangular or highly skewed shapes that are not realistic for real-world (Manhattan-world) buildings. The goal is to enforce strict orthogonality (axis-aligned horizontal and vertical edges) on the generated room polygons, even if it requires additional compute (making the model "heavier").

## Proposed Changes

We will introduce a **Test-Time Optimization (TTO)** mechanism coupled with an **Orthogonalization Post-Processing** step. This avoids having to retrain the entire diffusion model from scratch (which would take hours) while mathematically guaranteeing orthogonal outputs. 

### 1. Enable and Enhance Classifier-Free Guidance (`diffusion.py`)
The model already has a `_guide` function that uses gradients from auxiliary losses to shape the output during sampling. However, the orthogonal loss (`ortho_loss`) is currently missing from the guidance weights, and guidance is disabled by default.
- We will add `ortho=10.0` to the guidance weights in `_guide`.
- This step makes the model significantly "heavier" at inference time, as it unrolls gradient descent steps through the denoiser at each timestep to actively penalize non-orthogonal edges.

### 2. Active Contour Refinement (`generate.py`)
Even with guidance, the raw output might have slight imperfections before being converted to Shapely polygons. We will introduce a differentiable refinement step directly in `generate.py`:
- Before passing the coordinates to Shapely, we will instantiate a small Adam optimizer that runs for ~100 iterations on the raw coordinates.
- It will explicitly minimize a combined loss: `ortho_loss` (to square the edges) + `area_loss` (to prevent the polygon from collapsing into a line, a known failure mode of pure ortho_loss) + `validity_loss` (to prevent self-intersections).

### 3. Final Orthogonal Snapping (`generate.py`)
To mathematically guarantee that no triangular shapes remain, we will add a `force_orthogonal_polygon(poly)` function to the Shapely post-processing pipeline. 
- This function will extract the vertices of the heavily refined polygons and snap the sequence of edges to perfectly alternate between horizontal (Y=constant) and vertical (X=constant). 
- Since the TTO step will have already pushed the polygon to be ~99% orthogonal, this final snap will be a micro-adjustment that guarantees 100% realism without distorting the area or topology.

## User Review Required
> [!IMPORTANT]
> The addition of Test-Time Optimization (TTO) and enhanced guidance will significantly increase the inference time (making generation slower/heavier). However, it will mathematically guarantee the removal of triangular shapes without requiring a full model retraining. Do you approve of increasing inference time to achieve this?

## Verification Plan
### Automated Tests
Run the evaluation script on a small batch of test plans to verify that the generated images no longer contain triangular shapes and the `csr` (validity) metrics remain high:
```bash
python -m floorplan_diffusion.evaluate --ckpt checkpoints/phase3/best.pt --split test --k 16 --guidance 2.0 --steps 50 --plot 10 --n 10
```

### Manual Verification
The user should visually inspect the output images in `checkpoints/phase3/plots/` to confirm that the generated floorplans are strictly orthogonal and realistic.
