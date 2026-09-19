from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import tempfile

from .schema import ApartmentState
from .state import get_state, update_state, reset_state
from .llm_client import extract_requirements
from .validator import is_state_complete
from .graph import build_graph_report, graph_to_dict, render_graph

app = FastAPI(title="Phase 1 Demo - Requirement Extraction")

# Mount frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/")
def read_root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    current_state = get_state(req.session_id)
    new_state, error_msg = await extract_requirements(req.message, current_state)
    update_state(req.session_id, new_state)
    
    is_complete = is_state_complete(new_state)
    
    return {
        "state": new_state.model_dump(),
        "is_complete": is_complete,
        "message": error_msg if error_msg else "State updated successfully."
    }

@app.post("/reset/{session_id}")
def reset_endpoint(session_id: str = "default"):
    reset_state(session_id)
    return {"message": "State reset."}

@app.get("/graph/{session_id}")
def graph_endpoint(session_id: str = "default"):
    """
    Deterministically convert the current apartment state into the bubble
    constraint graph (PHASE_1.md section 6). This never calls the LLM --
    it only depends on whatever state has already been extracted.
    """
    current_state = get_state(session_id)
    G, warnings = build_graph_report(current_state)
    return {
        "graph": graph_to_dict(G),
        "warnings": warnings
    }

@app.get("/graph/{session_id}/render")
def graph_render_endpoint(session_id: str = "default"):
    """Render the current session's bubble diagram as a PNG image."""
    current_state = get_state(session_id)
    G, _ = build_graph_report(current_state)

    tmp_path = os.path.join(tempfile.gettempdir(), f"bubble_diagram_{session_id}.png")
    render_graph(G, tmp_path, title=f"Bubble Diagram — {session_id}")
    return FileResponse(tmp_path, media_type="image/png")