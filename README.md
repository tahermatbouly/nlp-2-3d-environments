# Prompt-Driven Architectural Design

**Integrating LLM Extraction, Diffusion-Based Floor Plan Generation, and Semantic Property Matching**

Graduation Research Project — Faculty of Computer Science, MSA University

> Karim Slama Elbana · Taher Mohamed Elmatbouly · Tamer M. Nassef

---

## Overview

Floor plan design traditionally requires either professional architectural expertise or rigid, pre-structured inputs such as bubble diagrams — limiting accessibility for non-expert users. This project proposes an end-to-end pipeline that lets a user describe an apartment or house in **natural language** (via Egyptian Arabic speech) and receive a generated, navigable **3D floor plan** in return.

```
User requirements → structured requirements → 2D floor plan → 3D apartment
```

The system is built around three sequential, independently testable phases, connected by two deterministic JSON contracts:

| Phase | Responsibility | Input → Output |
|---|---|---|
| **1. Requirement Extraction** | Convert free-form spoken/natural-language requirements into a structured, machine-readable form | Egyptian Arabic speech → structured JSON → bubble constraint graph |
| **2. Floor Plan Generation** | Generate a geometrically valid 2D floor plan from the structured requirements | Bubble constraint graph → 2D vector floor plan → geometry JSON |
| **3. 3D Reconstruction** | Deterministically reconstruct an interactive 3D model from the 2D geometry | Geometry JSON → extruded 3D model → GLB export |

This separation keeps language understanding, layout generation, and geometric reconstruction as distinct, swappable components — each can be developed, tested, and evaluated independently while a continuous data flow connects user intent to the final architectural model.

## System Architecture

```
┌─────────────────────────┐
│  PHASE 1                 │
│  Requirement Extraction   │
│                           │
│  Egyptian Arabic Speech   │
│         ↓ STT             │
│  Conversational LLM       │
│         ↓                 │
│  Structured English JSON  │
│         ↓ Python parser   │
│  Bubble Constraint Graph  │
└────────────┬──────────────┘
             ↓
┌─────────────────────────┐
│  PHASE 2                 │
│  Floor Plan Generation    │
│                           │
│  ChatHouseDiffusion       │
│  (Graphormer + Diffusion) │
│         ↓                 │
│  2D Vector Floor Plan     │
│         ↓                 │
│  Walls / Doors JSON       │
└────────────┬──────────────┘
             ↓
┌─────────────────────────┐
│  PHASE 3                 │
│  2D-to-3D Reconstruction  │
│                           │
│  Deterministic 3D Parser  │
│         ↓                 │
│  Wall Extrusion            │
│         ↓                 │
│  Interactive 3D Model      │
│         ↓                 │
│  GLB Export                 │
└─────────────────────────┘
```

*See [`paper/overleaf/figures/`](paper/overleaf/figures/) for the publication-ready version of this diagram.*

## Datasets

| Dataset | Role | Notes |
|---|---|---|
| **RPLAN** | Floor plan generation (Phase 2) | Structured residential floor-plan data; basis for ChatHouseDiffusion training |
| **Tell2Design** | Language-guided generation, evaluation | ~80,000 floor plans built from RPLAN with paired natural-language descriptions covering semantic, geometric, and topological information; 8 room categories |
| **Structured3D** | 3D reconstruction analysis (Phase 3) | ~3,500 professionally designed scenes with ground-truth 3D structural annotations (3,000 / 250 / 250 train-val-test split) |

RPLAN and Tell2Design are not independent — Tell2Design was constructed from RPLAN floor plans with added natural-language annotations, which makes it the primary evaluation ground for the requirement-extraction stage described above.

## Repository Structure

```
prompt-driven-floorplan/
├── config/                             # Pipeline & phase-specific configuration
├── data/
│   ├── raw/                            # RPLAN, Tell2Design, Structured3D
│   ├── processed/                      # Cleaned, split datasets
│   └── schemas/                        # JSON Schema contracts between phases
├── src/
│   ├── phase1_requirement_extraction/  # STT, conversational LLM, JSON→graph parser
│   ├── phase2_floorplan_generation/    # Graphormer + diffusion model, sampling, geometry extraction
│   ├── phase3_reconstruction/          # Wall extrusion, GLB export, 3D viewer
│   ├── future_extensions/              # Conversational editing & semantic property matching (planned)
│   ├── pipeline/                       # End-to-end orchestration + API
│   └── common/                         # Shared utilities
├── evaluation/                         # Metrics (Micro/Macro-IoU, graph consistency) & per-phase eval
├── scripts/                            # Dataset preprocessing, demo runners
├── notebooks/                          # Exploratory analysis
├── tests/                              # Unit & end-to-end tests
└── paper/                              # IEEE LaTeX source (Overleaf) and draft materials
```

Each phase directory is self-contained and can be run or tested against a hand-crafted input JSON without requiring the upstream phase to be implemented — see [`data/schemas/`](data/schemas/) for the exact contracts.

## Data Contracts

Two JSON schemas define the interfaces between phases:

- **`structured_requirements.schema.json` → `bubble_graph.schema.json`** — the output of the conversational LLM (Phase 1) and its deterministic conversion into the graph consumed by the diffusion model (Phase 2).
- **`floorplan_geometry.schema.json` → `glb_export.schema.json`** — the room/wall/door coordinate output of the diffusion model (Phase 2) and its conversion into a renderable 3D asset (Phase 3).

Keeping these as explicit, versioned schemas means changes to the visualization layer or the generative model never require touching the other end of the pipeline.

## Project Status

| Component | Status |
|---|---|
| Project concept & 3-phase pipeline | ✅ Established |
| Dataset roles (RPLAN, Tell2Design, Structured3D) | ✅ Established |
| Materials & Methods draft | ✅ Written |
| System architecture diagram | ✅ Designed |
| Repository scaffolding | ✅ Complete |
| Dataset splits & preprocessing | ⏳ In progress |
| Phase 1 implementation | ⏳ Not started |
| Phase 2 implementation | ⏳ Not started |
| Phase 3 implementation | ⏳ Not started |
| Conversational editing | ⚠️ Scoped, not confirmed for final system |
| Semantic property matching | ⚠️ Scoped, not confirmed for final system |
| Evaluation & results | ⏳ Not started |

## Methodology & References

This project builds on and extends:

- **ChatHouseDiffusion** (Qin et al., 2024) — prompt-guided diffusion-based floor plan generation and editing, using Graphormer for topological encoding and T5 for text conditioning.
- **HouseDiffusion** (Shabani et al., 2022) — vector floor plan generation conditioned on bubble diagrams.

Novel contributions targeted by this work:

1. An LLM-driven natural-language requirement extraction pipeline for floor plan generation (including a spoken Egyptian Arabic interface).
2. Movement beyond coarse categorical spatial attributes toward real-world measurable units.
3. Extension of the generation pipeline toward 3D reconstruction and (pending scope confirmation) conversational editing and semantic property matching.

Full reference list is maintained in [`paper/overleaf/main.tex`](paper/overleaf/main.tex).

## Authors

- **Karim Slama Elbana** — karim.salama@msa.edu.eg
- **Taher Mohamed Elmatbouly** — taher.mohamed1@msa.edu.eg
- **Tamer M. Nassef** (Supervisor) — tnassef@msa.edu.eg

Faculty of Computer Science, MSA University, 6th of October City, Giza, Egypt

## License

To be determined.