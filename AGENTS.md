# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a graduation research project implementing an end-to-end pipeline for generating 3D floor plans from natural language descriptions (Egyptian Arabic speech). The system consists of three phases:
1. **Requirement Extraction**: Convert speech to structured JSON and then to a bubble constraint graph.
2. **Floor Plan Generation**: Generate a 2D vector floor plan from the bubble constraint graph using a diffusion model.
3. **3D Reconstruction**: Deterministically reconstruct an interactive 3D model from the 2D geometry.

Currently, the repository contains the implementation of Phase 1 in the `phase1-demo` directory.

## Repository Structure

```
.
├── phase1-demo/                # Current implementation of Phase 1
│   ├── app/                    # FastAPI backend and core logic
│   │   ├── main.py             # Entry point: FastAPI application
│   │   ├── llm_client.py       # LLM integration (via Groq) for requirement extraction
│   │   ├── schema.py           # Pydantic models for apartment state
│   │   ├── state.py            # Session-based state management
│   │   ├── graph.py            # Bubble constraint graph generation
│   │   ├── layout.py           # Deterministic floor plan layout generation
│   │   ├── validator.py        # State validation
│   │   ├── prompts.py          # Prompt templates for LLM
│   │   └── speech/             # Speech-to-text module
│   ├── frontend/               # Static frontend served by the backend
│   │   └── index.html          # Chat interface with bubble diagram and floor plan visualization
│   ├── notebooks/              # Jupyter notebooks for exploration
│   ├── tests/                  # Unit tests
│   ├── requirements.txt        # Python dependencies
│   └── test_ollama.py          # Script for testing Ollama integration
├── .env                        # Environment variables (e.g., API keys)
├── README.md                   # High-level project overview (see repository root)
├── PHASE_1.md                  # Detailed description of Phase 1
└── PROGRESS_REPORT.md          # Progress tracking
```

## Development Setup

1. **Install dependencies**:
   ```bash
   cd phase1-demo
   pip install -r requirements.txt
   ```

2. **Set environment variables**:
   Copy `.env.example` to `.env` (if exists) or create `.env` with necessary variables:
   - `GROQ_API_KEY`: For accessing the LLM via Groq
   - Other API keys as needed

3. **Run the application**:
   ```bash
   cd phase1-demo
   uvicorn app.main:app --reload
   ```
   The API will be available at `http://localhost:8000`. The frontend is served at the root URL.

4. **Run tests**:
   ```bash
   cd phase1-demo
   pytest tests/
   ```
   Individual test files can be run with `pytest tests/test_*.py`.

## Common Commands

- **Start the development server**: `uvicorn app.main:app --reload` (from `phase1-demo/`)
- **Run all tests**: `pytest tests/` (from `phase1-demo/`)
- **Run a specific test**: `pytest tests/test_graph.py` (from `phase1-demo/`)
- **Check code formatting**: (Not configured; consider adding `ruff` or `black` if needed)
- **Install new dependencies**: Add to `requirements.txt` and run `pip install -r requirements.txt`

## Architecture Notes

### Phase 1 (Current Implementation)

The backend (`app/main.py`) provides the following endpoints:
- `GET /`: Serves the frontend HTML page.
- `POST /chat`: Processes text messages to extract requirements and update session state.
- `POST /voice-chat`: Processes audio input (via speech-to-text) and extracts requirements.
- `GET /graph/{session_id}`: Returns the current bubble constraint graph as JSON.
- `GET /layout/{session_id}`: Returns a deterministic floor plan layout (rectangular rooms) based on the current state.
- `GET /graph/{session_id}/render`: Returns a PNG rendering of the bubble diagram.
- `POST /reset/{session_id}`: Resets the session state.

The frontend (`frontend/index.html`) provides a chat interface with:
- Text input and voice recording (via Web Speech API).
- Real-time visualization of the bubble diagram (using vis.js).
- Real-time preview of the generated floor plan (simple rectangular layout).
- Display of the current extracted state as JSON.

Data flows as follows:
1. User speaks or types a description of an apartment.
2. Audio is transcribed to text (if voice input) or used directly.
3. The LLM (via Groq) extracts structured requirements (number of rooms, types, sizes, adjacencies).
4. The structured requirements are converted into a bubble constraint graph (nodes = rooms, edges = adjacency constraints).
5. A deterministic layout algorithm assigns rectangular positions to rooms that satisfy the constraints.
6. The bubble diagram and floor plan are rendered in the frontend.

### Future Phases

- **Phase 2**: Replace the deterministic layout with a diffusion-based model (ChatHouseDiffusion) to generate more realistic floor plans.
- **Phase 3**: Extrude the 2D floor plan into a 3D model and export as GLB.

## Code Conventions

- Follow the existing code style in the repository.
- Use type hints (via Pydantic and Python annotations).
- Keep functions focused and well-documented.
- For backend changes, ensure corresponding tests are updated or added.
- Frontend JavaScript should be kept readable; consider using modules if complexity increases.

## Notes

- The project uses session-based state (in-memory) for simplicity. For production, consider a more robust state store.
- The LLM client is configured to use Groq's API; switching providers requires changes in `llm_client.py`.
- The frontend relies on CDN-loaded libraries (vis.js, Google Fonts). For offline use, consider bundling dependencies.