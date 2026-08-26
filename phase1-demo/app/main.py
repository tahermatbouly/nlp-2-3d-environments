from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from .schema import ApartmentState
from .state import get_state, update_state, reset_state
from .ollama import extract_requirements
from .validator import is_state_complete

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
