# Progress Report — What Has Been Done So Far

**Project:** Prompt-Driven Architectural Design  
**Subtitle:** Integrating LLM Extraction, Diffusion-Based Floor Plan Generation, and Semantic Property Matching  

**Institution:** Faculty of Computer Science, MSA University  
**Authors:** Karim Slama Elbana · Taher Mohamed Elmatbouly  
**Supervisor:** Tamer M. Nassef  

**Report date:** 22 September 2026  
**Repository branch (active):** `nlp`

---

## 1. Project Goal (Recap)

The project aims to build an end-to-end pipeline that turns natural-language apartment requirements into a navigable 3D floor plan:

```
User requirements → structured JSON → 2D floor plan → 3D apartment
```

Work is organized into three independently testable phases:

| Phase | Responsibility | Planned output |
|---|---|---|
| **1. Requirement Extraction** | NL text (eventually Egyptian Arabic speech) → structured JSON → bubble constraint graph | Machine-readable apartment requirements |
| **2. Floor Plan Generation** | Bubble graph → geometrically valid 2D plan (ChatHouseDiffusion / Graphormer + diffusion) | Walls / doors geometry JSON |
| **3. 2D→3D Reconstruction** | Deterministic extrusion of 2D geometry | Interactive 3D model / GLB |

---

## 2. Executive Summary

**Phase 1 is largely implemented as a working demo.** Users can chat in English or Arabic, extract structured apartment requirements via a hosted LLM (Groq), view a live JSON state, see a bubble constraint graph, and preview a prototype rectangular 2D layout in the browser.

Phase 2 and Phase 3 are **not yet integrated into the application**. Early Phase-2-adjacent research exists in a Colab-style notebook (Tell2Design exploration and DDPM floor-plan image pretraining). Speech-to-text remains out of scope for the current demo.

---

## 3. Completed Work

### 3.1 Phase 1 — Working Demo (`phase1-demo/`)

A runnable demo was delivered with FastAPI backend and a single-page frontend.

**Pipeline in the demo today:**

```
User text → Groq LLM extraction → ApartmentState (JSON)
         → completeness validation
         → NetworkX bubble graph (+ PNG render)
         → prototype 2D rectangular layout
```

#### Backend (`phase1-demo/app/`)

| Module | What was done |
|---|---|
| `main.py` | FastAPI app: chat, reset, graph, layout, and static UI serving |
| `schema.py` | Pydantic models for rooms, requirements, and apartment state |
| `prompts.py` | Strict extraction prompt; English + Egyptian Arabic support; no geometry invention |
| `llm_client.py` | Active LLM path via Groq (`openai/gpt-oss-20b`), JSON cleaning/normalization |
| `state.py` | In-memory session state keyed by `session_id` |
| `validator.py` | Completeness check (≥1 bedroom, bathroom, kitchen, living room) |
| `graph.py` | Deterministic JSON → NetworkX bubble graph + PNG rendering |
| `layout.py` | Deterministic first spatial prototype: graph → rectangular room placements |
| `ollama.py` | Legacy local Ollama path (kept; not used by the active app) |

**APIs available:**

- `POST /chat` — extract requirements and update session state  
- `POST /reset/{session_id}` — clear session  
- `GET /graph/{session_id}` — bubble graph nodes/edges  
- `GET /graph/{session_id}/render` — PNG bubble diagram  
- `GET /layout/{session_id}` — rooms with `x, y, width, height`  
- `GET /` — serves the demo UI  

#### Frontend (`phase1-demo/frontend/`)

- Chat interface (send / reset, loading state, timeout handling)
- Live JSON state display
- Completion badge (Incomplete / Complete)
- Interactive bubble diagram (vis-network)
- Floor-plan preview from layout API (CSS-positioned room rectangles + room-type color legend)

#### Tests & examples

- Graph tests (`tests/test_graph.py`)
- Layout tests (`tests/test_layout.py`)
- LLM client smoke test (`tests/test_llm_client.py`)
- Sample incremental chat prompts (`examples/sample_requests.txt`)

#### Dependencies (`requirements.txt`)

FastAPI, Uvicorn, Pydantic, httpx, groq, NetworkX, Matplotlib.

---

### 3.2 NLP / Extraction Improvements

- Migrated extraction from **local Ollama** to a **hosted Groq model** for speed and reliability.
- Expanded the extraction prompt to accept **English, Arabic, and Egyptian Arabic** and normalize into the English schema.
- Constrained the LLM to **information extraction only** (explicit adjacencies; no invented geometry).

---

### 3.3 Prototype 2D Layout (beyond original Phase-1 scope)

In addition to the bubble graph target for Phase 1, a **deterministic rectangular layout prototype** was added:

- Maps room size labels to placeholder dimensions
- Places connected rooms using a simple graph-based placement
- Exposed via API and shown in the frontend as a floor-plan preview

This is a first spatial prototype — **not** the planned ChatHouseDiffusion generative model.

---

### 3.4 Research / Experimental Work (Notebook)

`phase1-demo/notebooks/Grad (2).ipynb` explores Phase-2-adjacent topics:

- Tell2Design dataset download and parsing
- Caption → room keyword / graph statistics
- Pairing floor-plan images with graphs
- Pretraining a DDPM `UNet2DModel` (Hugging Face Diffusers) on floor-plan images
- Sampling smoke tests

This work is **experimental and not wired into the FastAPI demo**.

---

### 3.5 Project Documentation & Scaffolding

| Artifact | Role |
|---|---|
| `README.md` | Full research vision, 3-phase architecture, datasets, methodology |
| `PHASE_1.md` | Concrete Phase-1 demo specification (schema, API, graph) |
| Git history | Scaffolding → first working demo → frontend polish → Groq + graph + Arabic prompts → notebook → 2D layout |

**Main development themes (from commits):**

1. Repository init and folder structure  
2. First working Phase-1 demo + frontend improvements  
3. NLP architecture update (Groq, graph module, independent tests)  
4. Arabic / Egyptian Arabic extraction support  
5. Tell2Design / diffusion notebook  
6. Prototype 2D floor-plan layout + UI preview  

---

## 4. Current Capability Snapshot

| Capability | Status |
|---|---|
| Project concept & 3-phase design docs | Done |
| Phase 1 text → structured JSON (LLM) | Done (Groq) |
| Multilingual EN / AR extraction | Done |
| Session state + completion check | Done |
| Bubble constraint graph (deterministic) | Done |
| Bubble PNG render API | Done |
| Prototype 2D rectangular layout | Done (prototype only) |
| Interactive web demo | Done |
| Graph / layout automated tests | Done |
| Tell2Design + image DDPM exploration | Experimental (notebook) |
| ChatHouseDiffusion / Phase 2 product path | Not started in app |
| 3D reconstruction / GLB (Phase 3) | Not started |
| Speech interface (STT / TTS) | Not started |
| Full monorepo scaffolding (`config/`, `data/`, `src/`, `evaluation/`, `paper/`) | Documented only |

---

## 5. What Remains / Next Steps

### Near term (Phase 1 polish)

- Align root `README.md` project-status table with the implemented demo  
- Clean up legacy Ollama paths / broken root smoke-test imports  
- Strengthen layout rules (collision handling, more realistic adjacency placement)  
- Formalize versioned JSON Schema contracts under `data/schemas/` (as planned in the README)

### Phase 2 — Floor Plan Generation

- Integrate a real generative floor-plan model (e.g. ChatHouseDiffusion / Graphormer + diffusion), replacing the rectangular prototype  
- Train / evaluate on RPLAN / Tell2Design  
- Produce walls/doors geometry JSON as the Phase-2 contract

### Phase 3 — 3D Reconstruction

- Deterministic wall extrusion from 2D geometry  
- Interactive 3D viewer and GLB export

### Broader project goals

- Egyptian Arabic speech input (STT)  
- Evaluation metrics and per-phase evaluation harness  
- Optional future extensions: conversational editing, semantic property matching  

---

## 6. Tech Stack (Used So Far)

| Layer | Technologies |
|---|---|
| Language | Python 3 |
| API | FastAPI, Uvicorn, Pydantic v2 |
| LLM (current) | Groq SDK |
| LLM (legacy) | Ollama |
| Graphs / viz (backend) | NetworkX, Matplotlib |
| Frontend | HTML / CSS / JS, vis-network |
| Notebook ML | PyTorch, Diffusers (DDPM / UNet2D), pandas, PIL |
| Secrets | `.env` (`GROQ_API_KEY`) |

---

## 7. Conclusion

To date, the team has delivered a **functional Phase-1 requirement-extraction demo** with multilingual LLM extraction, deterministic bubble-graph generation, a first 2D layout prototype, an interactive UI, and supporting tests. Parallel notebook work has begun exploring diffusion-based floor-plan generation. The main remaining effort is implementing production Phase-2 generative layout and Phase-3 3D reconstruction, then connecting them through the planned JSON contracts into a full end-to-end system.
