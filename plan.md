## Goal Description
The user requested three specific constraints to improve floorplan realism:
1. Ensure all outer rooms (rooms other than the "living" room) are perfectly rectangular (exactly 4 corners).
2. Ensure rooms are not excessively elongated (aspect ratio limit).
3. Set a strict lower bound on room area to prevent unrealistically small rooms.

## Proposed Changes

### `floorplan_diffusion/diffusion.py`
1. **Four Corners Constraint**: In the `sample` function, we intercepted the token count mask. If the room is not a living room, we force the mask to have exactly `4` vertices. Because our pipeline already orthogonalizes the shapes, a 4-vertex orthogonal polygon is mathematically guaranteed to be a perfect rectangle.
2. **Aspect Ratio Weight**: Added a new weight configuration `w_aspect` to correctly pass aspect ratio penalties through the `_guide` step.

### `floorplan_diffusion/losses.py`
1. **Aspect Ratio Penalty**: Added a new differentiable `aspect_ratio_loss` function that computes the bounding box aspect ratio of each room. It heavily penalizes shapes where the ratio of the longest to shortest side exceeds `3.0`. We added this penalty to the global `aux_losses` dictionary.

### `floorplan_diffusion/generate.py`
1. **Minimum Area Clamp**: Right after constructing the batch for sampling, we clamp the target `area` of every room to a minimum of `20.0` units. This forcibly conditions the network (and the downstream Test-Time Optimization) to expand tiny rooms up to a realistic size limit.
2. **Aspect Ratio Refinement**: Added the `aspect_ratio_loss` to the Active Contour TTO loop with a strong weight (`5.0`), dynamically reshaping elongated rectangles into squarer proportions during inference.

## User Review Required
> [!IMPORTANT]
> The minimum area limit is now hardcoded to `20.0` squared units, and the maximum aspect ratio is set to `3.0`. By forcing outer rooms to have exactly 4 vertices, their complexity will be significantly reduced, but they are guaranteed to be perfect rectangles. Are you okay with these specific limit values?

## Verification Plan
### Manual Verification
After implementation, you can run `python inference.py --ckpt checkpoints/smoke_phase3/best.pt --guidance 2.0 --device cpu` and verify that the generated `inference_out.png` shows perfect rectangles for non-living rooms, no extremely thin corridors, and no tiny unreadable rooms.
