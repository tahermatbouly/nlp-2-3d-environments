## Goal Description
Currently, the generated floorplans display rooms directly touching each other, with the `wall` layer only tracing the exterior hull of the building. This makes it hard to visually distinguish individual rooms and see the doors connecting them. The goal is to generate clear interior walls separating all rooms and cut out clear openings for the procedural doors.

## Proposed Changes

### `floorplan_diffusion/generate.py`
We will modify the `to_plan_dict` function, which is responsible for converting the raw model outputs into the final drawable layers:
1. **Interior Walls**: Instead of buffering only the `unary_union` (which dissolves internal boundaries), we will extract the `boundary` of every individual room and buffer them by `1.5` units. This creates a solid wall network separating every room.
2. **Door Openings**: We will subtract the procedural `door` geometries from this new wall network. Since the walls are 3.0 units thick (1.5 on each side) and the doors are 4.0 units thick, subtracting the doors will create clean, realistic gaps in the walls.

#### [MODIFY] floorplan_diffusion/generate.py
```python
    # Procedural Walls and Inner boundary
    allg = [d["geometry"] for _, d in cand.graph.nodes(data=True) if not d["geometry"].is_empty]
    if allg:
        inner = unary_union([g if g.is_valid else make_valid(g) for g in allg])
        plan["inner"] = inner
        try:
            # Create walls along all boundaries (interior + exterior)
            boundaries = unary_union([g.boundary for g in allg if g.is_valid])
            wall = boundaries.buffer(1.5, cap_style=2, join_style=2)
            
            # Subtract doors from the wall to create clear openings
            if "door" in plan and not plan["door"].is_empty:
                wall = wall.difference(plan["door"])
                
            if not wall.is_empty:
                plan["wall"] = wall
        except Exception:
            pass
```

## User Review Required
> [!IMPORTANT]
> The walls will now have a thickness of 3.0 canvas units (1.5 on each side of the boundary). Procedural doors will carve 4.0-unit gaps through them. This will make the visual output match standard architectural styles. Do you approve these changes?

## Verification Plan
### Manual Verification
After implementation, the user can run `python inference.py --ckpt checkpoints/phase3/best.pt --guidance 2.0 --out inference_out.png` and visually verify that `inference_out.png` displays thick yellow interior walls separating the rooms, with clean pink doors cutting through them.
