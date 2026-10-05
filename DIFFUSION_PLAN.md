# Graph-Conditioned Geometric Diffusion Model for Floorplan Generation

## 1. Problem Definition

The model takes a **floorplan graph** as input and generates the geometric layout of every room.

### Input

A NetworkX graph where:

* **Nodes** represent rooms.
* Each node contains:

  * `room_type`
  * `area`
* **Edges** represent relationships between rooms.
* Each edge contains:

  * `connection_type`

Example:

```text
Bedroom ── door ── Living ── opening ── Kitchen
   │
  door
   │
Bathroom
```

### Output

For every room:

* Room type
* Area
* Ordered set of polygon corner coordinates

Example:

```text
Bedroom:
[(1.2, 2.1), (4.2, 2.1), (4.2, 5.0), (1.2, 5.0)]
```

The neural network does **not** directly output NetworkX or Shapely objects.

It outputs numerical tensors.

After generation:

```text
Coordinates
    ↓
Shapely Polygon
    ↓
NetworkX Graph
```

---

# 2. Overall Architecture

The proposed architecture is:

```text
                    INPUT GRAPH
                        │
                        ▼
               ┌─────────────────┐
               │ Graph Transformer│
               └─────────────────┘
                        │
                 Room embeddings
                        │
                        ▼
          ┌─────────────────────────┐
          │ Conditional Diffusion   │
          │                         │
          │ Noisy coordinates       │
          │ + graph embeddings      │
          │ + timestep              │
          └────────────┬────────────┘
                       │
                       ▼
                Polygon coordinates
                       │
                 Geometry validation
                       │
                       ▼
                 Shapely Polygons
                       │
                       ▼
                  NetworkX Graph
```

The central idea is:

> **The Graph Transformer understands the rooms and their relationships, while the diffusion model learns where those rooms can physically be placed.**

---

# 3. Data Representation

## 3.1 Graph Representation

Each floorplan is represented as a NetworkX graph.

### Node attributes

Each node should contain:

```text
room_type
area
```

### Edge attributes

Each edge should contain:

```text
connection_type
```

Examples:

```text
door
opening
adjacency
```

The exact connection types should be defined once and remain consistent throughout the dataset.

---

# 4. Geometry Representation

Each room is represented as a polygon.

Example:

```text
[(x1, y1),
 (x2, y2),
 (x3, y3),
 (x4, y4)]
```

The coordinates remain in **meters** in the original dataset.

The model, however, should work with normalized coordinates.

---

# 5. Geometry Preprocessing

Every polygon must go through the same preprocessing pipeline before being used for training.

```text
Shapely Polygon
      ↓
Validate
      ↓
Remove duplicate points
      ↓
Remove unnecessary collinear points
      ↓
Fix orientation
      ↓
Fix starting vertex
      ↓
Store ordered coordinates
```

## Rules

### Rule 1 — Every polygon must be valid

No:

* self-intersections
* empty geometries
* malformed polygons

Invalid samples should be fixed if reliably possible or removed from training.

### Rule 2 — Remove redundant vertices

If a vertex lies directly on a straight edge:

```text
A ─ B ─ C
```

and `B` is unnecessary, store:

```text
A ─ C
```

### Rule 3 — Use consistent orientation

All polygons must use the same orientation.

For example:

```text
Counter-clockwise
```

### Rule 4 — Use deterministic vertex ordering

The same polygon must always produce the same coordinate sequence.

For example:

```text
A → B → C → D
```

should not sometimes appear as:

```text
C → D → A → B
```

This is important because the diffusion model sees polygon coordinates as numerical data.

---

# 6. Variable Number of Polygon Vertices

Different rooms may have different numbers of corners.

Example:

```text
Bedroom → 4 corners
Kitchen → 6 corners
Living   → 8 corners
```

Therefore the model must support variable-length polygons.

## Recommended representation

Choose a maximum number of vertices:

```text
MAX_VERTICES = V
```

Represent every room as:

```text
V × 2
```

where each row is:

```text
(x, y)
```

Unused positions are padded.

A separate mask identifies valid vertices.

Example:

```text
Coordinates:
[(x1,y1),
 (x2,y2),
 (x3,y3),
 (x4,y4),
 (0,0),
 (0,0)]

Mask:
[1, 1, 1, 1, 0, 0]
```

## Additional recommendation

Use a separate **vertex-count prediction mechanism**.

The model should learn:

```text
Bedroom → 4 vertices
Kitchen → 6 vertices
Living → 8 vertices
```

This prevents the model from having to infer where the polygon ends purely from coordinate values.

---

# 7. Coordinate Normalization

Do not train directly on arbitrary metric coordinates.

First remove global translation.

For example:

```text
Floorplan centroid → (0, 0)
```

Then normalize using a **fixed dataset-wide scale**.

For example:

```text
normalized_x = x / S
normalized_y = y / S
```

where `S` is determined from the training dataset.

After generation:

```text
x = normalized_x × S
y = normalized_y × S
```

## Important rule

Do **not** independently normalize every floorplan to `[0,1] × [0,1]` unless scale invariance is explicitly desired.

Doing that removes useful information about the physical size of the floorplan.

Translation can be removed because it does not affect the architectural layout.

---

# 8. Graph Encoder

Use a **Graph Transformer** as the graph encoder.

The Graph Transformer converts the input graph into an embedding for every room.

Conceptually:

```text
Graph
  ↓
Graph Transformer
  ↓
h₁, h₂, h₃, ..., hₙ
```

where each `hᵢ` represents the learned context of one room.

Each room embedding should contain information about:

* Its room type
* Its area
* Its neighboring rooms
* Connection types
* Indirect relationships to other rooms
* Global floorplan structure

---

# 9. Node Encoding

For every room:

### Categorical information

Encode:

```text
room_type
```

using a learnable embedding.

### Numerical information

Encode:

```text
area
```

after normalization.

The node representation becomes:

```text
Room embedding =
    room_type embedding
    +
    area representation
```

---

# 10. Edge Encoding

Connection types must be explicitly represented.

For example:

```text
door
opening
adjacency
```

should each have a learnable embedding.

The Graph Transformer should therefore know the difference between:

```text
Bedroom ── door ── Living
```

and:

```text
Bedroom ── adjacency ── Living
```

These relationships can imply different geometric constraints.

---

# 11. Global Floorplan Representation

Add a **global floorplan representation/token**.

This allows the model to reason about the graph as a whole rather than only individual rooms.

Conceptually:

```text
Room embeddings
     +
Global floorplan embedding
```

This is useful for learning:

* Overall arrangement
* Central vs peripheral rooms
* Typical structural layouts
* Global proportions

---

# 12. Diffusion Model

The diffusion model operates directly on the polygon coordinates.

Let:

```text
X₀
```

represent the real floorplan coordinates.

Training gradually adds noise:

```text
X₀ → X₁ → X₂ → ... → Xₜ
```

where higher `t` means more noise.

The model receives:

```text
Noisy coordinates
+
Diffusion timestep
+
Graph embeddings
```

and learns to recover the underlying geometry.

---

# 13. Diffusion Conditioning

The diffusion network must be conditioned on the graph.

Conceptually:

```text
Noisy coordinates
       │
       ├────────────┐
       ▼            │
Diffusion model     │
       ▲            │
       │            │
Graph embeddings ───┘
```

Each room's coordinates should be associated with its corresponding room embedding.

Each vertex can also receive:

* Vertex position/index embedding
* Diffusion timestep embedding
* Room embedding

---

# 14. Recommended Diffusion Architecture

Use:

```text
Graph Transformer
        ↓
Room embeddings
        ↓
Cross-attention / conditioning
        ↓
Vertex/Polygon Transformer
        ↓
Coordinate prediction
```

The Graph Transformer handles the graph.

The coordinate diffusion network handles the geometry.

---

# 15. Why Diffusion

A single input graph may have multiple valid layouts.

Example:

```text
Bedroom ─ Living ─ Kitchen
```

could produce multiple valid geometric arrangements.

A deterministic regression model tends to learn an average solution.

Diffusion instead learns:

> **the distribution of valid layouts conditioned on the graph.**

Therefore the same graph can generate multiple different floorplans.

```text
Same graph
    ↓
Random noise A → Layout A
Random noise B → Layout B
Random noise C → Layout C
```

---

# 16. Training Process

For every training floorplan:

```text
1. Load NetworkX graph
2. Extract room features
3. Extract edge features
4. Extract Shapely polygons
5. Preprocess polygons
6. Normalize coordinates
7. Determine vertex counts
8. Create coordinate tensor
9. Add diffusion noise
10. Encode the graph
11. Predict the underlying geometry
12. Calculate losses
13. Update model
```

---

# 17. Primary Training Objective

The main objective is the **diffusion coordinate loss**.

The model learns to denoise noisy coordinates toward the original floorplan geometry.

The core objective should remain the diffusion loss.

---

# 18. Auxiliary Losses

Do not rely solely on coordinate-level diffusion loss.

Add additional objectives where appropriate.

## 18.1 Area loss

Compare:

```text
Predicted room area
vs
Input room area
```

This encourages the generated polygon to preserve the required room size.

---

## 18.2 Geometry validity loss

Penalize malformed polygon predictions where possible.

Examples:

* Self-intersections
* Degenerate polygons
* Extremely small areas

---

## 18.3 Overlap loss

Rooms belonging to the same floorplan should generally not overlap.

Penalize excessive intersections between rooms.

---

## 18.4 Connectivity loss

For an edge such as:

```text
Bedroom ── door ── Living
```

the generated polygons should have spatial compatibility with a door connection.

---

## 18.5 Non-connectivity constraint

Rooms that should not be connected should not accidentally satisfy the geometric relationship associated with a connection.

---

# 19. Differentiable vs Exact Geometry

Do not depend entirely on Shapely during neural-network optimization.

Shapely operations are primarily useful for validation and post-processing.

Use:

```text
Differentiable geometric approximations
```

during training where necessary.

Then use:

```text
Exact Shapely geometry checks
```

after generation.

This separation keeps the training pipeline practical.

---

# 20. Inference Pipeline

At inference:

```text
Input NetworkX graph
        ↓
Graph Transformer
        ↓
Room embeddings
        ↓
Random coordinate initialization
        ↓
Diffusion denoising
        ↓
Generated coordinates
        ↓
Vertex count reconstruction
        ↓
Denormalize coordinates
        ↓
Shapely polygons
        ↓
Geometry validation
        ↓
Valid floorplan
```

---

# 21. Generate Multiple Candidates

Do not generate only one result.

For a given graph, generate multiple samples:

```text
Graph
 ↓
 ├── Sample 1
 ├── Sample 2
 ├── Sample 3
 ├── ...
 └── Sample K
```

Because diffusion is probabilistic, different initial noise produces different layouts.

---

# 22. Candidate Validation and Ranking

Every generated floorplan should go through a validation pipeline.

Check:

```text
✓ Correct number of rooms
✓ Correct room types
✓ Valid polygons
✓ Reasonable room areas
✓ Low room overlap
✓ Correct connectivity
✓ Valid edge relationships
✓ Reasonable overall dimensions
```

Invalid candidates should be rejected.

---

# 23. Candidate Scoring

Each candidate can receive a score such as:

```text
score =
    area_error
  + overlap_penalty
  + topology_penalty
  + invalid_geometry_penalty
  + compactness_penalty
```

The exact weighting should be determined experimentally.

The best valid candidate can then be selected.

---

# 24. Shapely Reconstruction

After diffusion finishes:

```text
Predicted coordinates
        ↓
Shapely Polygon
```

Each polygon becomes the geometry of its corresponding graph node.

The final graph becomes:

```text
NetworkX graph

Node:
    room_type
    area
    geometry = Shapely Polygon

Edge:
    connection_type
```

---

# 25. Coordinate System Rules

Keep the coordinate system consistent across the entire project.

### Dataset

```text
meters
```

### Neural network

```text
normalized metric coordinates
```

### Output

```text
denormalized meters
```

### Geometry library

```text
Shapely metric coordinates
```

Never mix coordinate systems without an explicit conversion.

---

# 26. Dataset Splitting

Split the dataset into:

```text
Train
Validation
Test
```

Do not allow duplicate or near-duplicate floorplans to exist across different splits.

Otherwise the model may effectively see the same floorplan during training and testing.

---

# 27. Evaluation Metrics

Do not evaluate this model using image-generation metrics.

The model should be evaluated as a **geometric generator**.

## Room-level metrics

Measure:

* Area error
* Centroid error
* Polygon IoU
* Coordinate error
* Vertex-count accuracy

## Floorplan-level metrics

Measure:

* Invalid polygon percentage
* Room overlap percentage
* Connectivity accuracy
* Topology consistency
* Overall geometric validity

## Most important metric

### Constraint Satisfaction Rate

Measure:

> Percentage of generated floorplans that satisfy all required constraints.

This should be one of the primary metrics.

---

# 28. Development Phases

Do not implement everything at once.

## Phase 1 — Rectangular rooms

Represent each room as:

```text
center_x
center_y
width
height
```

Train:

```text
Graph Transformer
        ↓
Conditional diffusion
        ↓
Rectangle parameters
```

This validates the fundamental idea.

---

## Phase 2 — Arbitrary polygons

Replace rectangle parameters with:

```text
N × (x, y)
```

and introduce:

```text
Vertex count
+
Vertex mask
```

Now the model can represent irregular rooms.

---

## Phase 3 — Geometric constraints

Add:

```text
Area constraints
Overlap constraints
Adjacency constraints
Connection constraints
```

---

## Phase 4 — Multiple sampling

Generate multiple layouts per graph.

Add candidate validation and ranking.

---

## Phase 5 — Shapely / NetworkX integration

Build the complete pipeline:

```text
NetworkX graph
      ↓
Model
      ↓
Coordinates
      ↓
Shapely polygons
      ↓
Validated floorplan graph
```

---

# 29. Non-Negotiable Rules

## Data rules

1. All coordinates must use meters.
2. Every polygon must be valid.
3. Remove duplicate vertices.
4. Remove redundant collinear vertices.
5. Every polygon must use the same orientation.
6. Every polygon must use deterministic vertex ordering.
7. Normalize coordinates consistently.
8. Never expose ground-truth coordinates to the graph encoder.

## Model rules

9. Use a Graph Transformer as the graph encoder.
10. Use continuous diffusion for coordinate generation.
11. Explicitly encode room type.
12. Explicitly encode room area.
13. Explicitly encode connection type.
14. Support variable polygon vertex counts.
15. Use masks for padded vertices.
16. Include diffusion timestep information.
17. Generate multiple samples at inference.

## Geometry rules

18. Convert model output to Shapely only after neural inference.
19. Reject invalid polygons.
20. Reject severe room overlap.
21. Check predicted area against requested area.
22. Check generated geometry against graph connectivity.

## Evaluation rules

23. Do not evaluate using only coordinate MSE.
24. Measure actual geometric validity.
25. Measure topology/connectivity correctness.
26. Measure constraint satisfaction.
27. Keep duplicates out of the test set.

---

# 30. Final System

The complete target system is:

```text
                 NetworkX Graph
                       │
                       │
          ┌────────────┴────────────┐
          │                         │
     Room features             Edge features
          │                         │
          └────────────┬────────────┘
                       ▼
               Graph Transformer
                       │
                Room embeddings
                       │
                       ▼
              Conditional Diffusion
                       │
               Random coordinates
                       │
                 Denoising steps
                       │
                       ▼
              Polygon coordinates
                       │
                       ▼
              Geometry validation
                       │
                       ▼
                 Shapely polygons
                       │
                       ▼
                NetworkX floorplan
```

## Core concept

> **Graph Transformer = understand the floorplan topology.**

> **Diffusion = generate the geometry.**

> **Shapely = represent and validate the geometry.**

> **NetworkX = represent the final floorplan graph.**

The neural model itself only needs to learn:

```text
Graph structure + room information
                ↓
       Distribution of valid
          room coordinates
```
