import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.agent.graph import agent_graph, manager
from backend.agent.state import new_state
from backend.session_store import SessionStore


BASE_DIR = Path(__file__).parent

logger = logging.getLogger("chat")

app = FastAPI(title="Floor plan requirements agent")

store = SessionStore(BASE_DIR / "data" / "sessions.db")

# One lock per chat: two quick messages must not run the graph on the
# same stale state at the same time.
_locks: dict[str, asyncio.Lock] = {}


def _lock_for(session_id: str) -> asyncio.Lock:
    return _locks.setdefault(session_id, asyncio.Lock())


# ============================================================
# Request models
# ============================================================

class CreateSession(BaseModel):
    language: str = "ar_eg"


class SendMessage(BaseModel):
    text: str = Field(..., max_length=2000)


# ============================================================
# Reply building
# ============================================================
# The graph returns segments (built for speech). For text chat we
# join them into one message, and keep the segments in the payload
# so "read aloud" and voice mode can reuse them later.

def _assistant_message(state: dict) -> dict:

    segments = state["intent_prompt"] + state["current_prompt"]

    return {
        "role": "assistant",
        "text": " ".join(seg["text"] for seg in segments),
        "segments": segments,
    }


def _quick_replies(state: dict) -> list[str]:
    """
    Tap-to-answer buttons when the agent asks ONE question that has
    a fixed set of answers. The button text is sent as a normal
    message, so the graph treats it like anything the user typed.
    """

    if state["mode"] not in ("gathering", "correcting"):
        return []

    if len(state["current_slots"]) != 1:
        return []

    slot = manager.get_slot(state["current_slots"][0])
    lang = state["language"]
    req = state["requirements"]

    if slot is None:
        return []

    if slot["type"] == "boolean":
        return [
            manager.text(manager.messages["yes"], lang),
            manager.text(manager.messages["no"], lang),
        ]

    if slot["type"] == "enum" and slot.get("options"):
        return [manager.option_name(option, lang) for option in slot["options"]]

    if slot["type"] == "enum" and slot.get("options_from") == "rooms_list":
        return [manager.room_label(req, r["id"], lang) for r in manager.rooms(req)]

    return []


def _payload(state: dict, assistant: dict) -> dict:

    return {
        "message": assistant,
        "quick_replies": _quick_replies(state),
        "mode": state["mode"],
        "done": state["conversation_finished"],
        "ended_early": state["ended_early"],
        "room_graph": state["room_graph"],
    }


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/sessions")
async def create_session(body: CreateSession = CreateSession()):
    """Start a chat. The agent speaks first (the greeting)."""

    if body.language not in manager.languages:
        raise HTTPException(422, f"language must be one of {manager.languages}")

    try:
        state = await agent_graph.ainvoke(new_state(body.language))
    except Exception:
        logger.exception("Graph failed while starting a session")
        raise HTTPException(502, "Could not start the chat. Please try again.")

    assistant = _assistant_message(state)
    session_id = uuid.uuid4().hex

    store.save(session_id, state, [assistant])

    return {"session_id": session_id, "language": state["language"], **_payload(state, assistant)}


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: SendMessage):
    """One user turn: text in, the agent's reply out."""

    text = body.text.strip()

    if not text:
        raise HTTPException(422, "Message is empty.")

    async with _lock_for(session_id):

        loaded = store.load(session_id)

        if loaded is None:
            raise HTTPException(404, "Unknown session.")

        state, messages = loaded

        if state["conversation_finished"]:
            raise HTTPException(409, "This conversation is finished. Start a new chat.")

        try:
            result = await agent_graph.ainvoke({**state, "transcript": text})
        except Exception:
            # Nothing is saved, so the user can simply resend.
            logger.exception("Graph failed on session %s", session_id)
            raise HTTPException(502, "The model request failed. Please try again.")

        assistant = _assistant_message(result)

        messages += [{"role": "user", "text": text}, assistant]

        store.save(session_id, result, messages)

    return _payload(result, assistant)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Reload a chat (page refresh)."""

    loaded = store.load(session_id)

    if loaded is None:
        raise HTTPException(404, "Unknown session.")

    state, messages = loaded

    return {
        "session_id": session_id,
        "language": state["language"],
        "messages": messages,
        "quick_replies": _quick_replies(state),
        "mode": state["mode"],
        "done": state["conversation_finished"],
        "ended_early": state["ended_early"],
        "room_graph": state["room_graph"],
    }


@app.get("/sessions/{session_id}/state")
async def get_state(session_id: str):
    """Debug view: what the agent has collected so far."""

    loaded = store.load(session_id)

    if loaded is None:
        raise HTTPException(404, "Unknown session.")

    state, _ = loaded

    return {
        "mode": state["mode"],
        "requirements": state["requirements"],
        "resolved": state["resolved"],
        "assumed": state["assumed"],
        "acknowledged_rules": state["acknowledged_rules"],
        "pending_validation": state["pending_validation"],
    }