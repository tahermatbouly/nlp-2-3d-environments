"""
Text chat API for the floor plan agent.

Put this file at  Backend/api/app.py
      store.py at Backend/api/store.py
      chat.html at Backend/static/chat.html

Run:  uvicorn Backend.api.app:app --reload
Open: http://localhost:8000

Voice (dictation, read aloud, voice mode) is not here yet. When the
speech models are chosen, they plug in as extra endpoints that end in
the same place: set state["transcript"] and call agent_graph.
"""

import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.agent.graph import agent_graph, manager
from backend.agent.state import new_state
from backend.store import SessionStore

logger = logging.getLogger("chat")

CHAT_PAGE = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

app = FastAPI(title="Floor plan assistant")
store = SessionStore(os.getenv("CHAT_DB_PATH", "chat.db"))

# One turn at a time per chat: a double-click or two tabs must not
# run two graph calls on the same state.
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


# ============================================================
# Request models
# ============================================================

class CreateSession(BaseModel):
    language: Literal["en", "ar_eg"] = "ar_eg"


class SendMessage(BaseModel):
    text: str = Field(..., max_length=1000)


# ============================================================
# Building the reply
# ============================================================

# Quick replies are what the user would say, so they read naturally
# when sent back as a normal message (the LLM extracts the answer).
QUICK_LABELS = {
    "auto": {"en": "Arrange them automatically", "ar_eg": "رتبها إنت أوتوماتيك"},
    "manual": {"en": "I'll specify them", "ar_eg": "هحددها أنا"},
    "hallway": {"en": "A hallway", "ar_eg": "طرقة"},
    "living": {"en": "The living room", "ar_eg": "الريسبشن"},
}

SUMMARY_CONFIRM = {"en": "Yes, that's right", "ar_eg": "أيوه كده تمام"}
WARNING_KEEP = {"en": "Keep it as it is", "ar_eg": "سيبها زي ما هي"}


def quick_replies(state: dict) -> list[str]:
    """Tap-to-answer buttons for the question currently on screen."""

    lang = state["language"]
    req = state["requirements"]
    mode = state["mode"]

    if state["conversation_finished"]:
        return []

    if mode == "summary":
        return [SUMMARY_CONFIRM[lang]]

    if mode == "validating":

        pending = state["pending_validation"]

        if pending["kind"] == "warning":
            return [WARNING_KEEP[lang]]

        return [
            manager.room_label(req, r["id"], lang)
            for r in manager.rooms(req)
            if r["id"] != pending["room"]
        ]

    if mode != "gathering" or len(state["current_slots"]) != 1:
        return []

    slot = manager.get_slot(state["current_slots"][0])

    if slot["type"] == "boolean":
        return [manager.text(manager.messages[k], lang) for k in ("yes", "no")]

    if slot["type"] == "enum" and slot.get("options"):
        return [
            manager.text(QUICK_LABELS[o], lang) if o in QUICK_LABELS else manager.option_name(o, lang)
            for o in slot["options"]
        ]

    if slot["type"] == "enum" and slot.get("options_from") == "rooms_list":
        return [manager.room_label(req, r["id"], lang) for r in manager.rooms(req)]

    return []


def build_reply(state: dict) -> dict:
    """
    What the client receives after a turn. `segments` is the same
    list a voice client would play, so read-aloud and voice mode can
    reuse this exact payload later.
    """

    segments = state["intent_prompt"] + state["current_prompt"]

    return {
        "text": " ".join(seg["text"] for seg in segments),
        "segments": segments,
        "mode": state["mode"],
        "done": state["conversation_finished"],
        "ended_early": state["ended_early"],
        "quick_replies": quick_replies(state),
        "room_graph": state["room_graph"],
    }


def _store_reply(session_id: str, reply: dict) -> None:

    store.add_message(
        session_id,
        "assistant",
        reply["text"],
        {"quick_replies": reply["quick_replies"], "done": reply["done"]},
    )


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def index():
    return FileResponse(CHAT_PAGE)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/sessions")
async def create_session(body: CreateSession):
    """Start a chat. The graph's first call plays the greeting."""

    try:
        state = await agent_graph.ainvoke(new_state(body.language))
    except Exception:
        logger.exception("Graph failed on session start")
        raise HTTPException(502, "The assistant is not available right now.")

    session_id = store.create(state)
    reply = build_reply(state)
    _store_reply(session_id, reply)

    return {"session_id": session_id, "language": body.language, "reply": reply}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Everything the page needs to restore a chat after a reload."""

    state = store.load_state(session_id)

    if state is None:
        raise HTTPException(404, "Session not found.")

    return {
        "session_id": session_id,
        "language": state["language"],
        "finished": state["conversation_finished"],
        "room_graph": state["room_graph"],
        "messages": store.history(session_id),
    }


@app.post("/sessions/{session_id}/messages")
async def send_message(session_id: str, body: SendMessage):

    text = body.text.strip()

    if not text:
        raise HTTPException(422, "Message is empty.")

    async with _locks[session_id]:

        state = store.load_state(session_id)

        if state is None:
            raise HTTPException(404, "Session not found.")

        if state["conversation_finished"]:
            raise HTTPException(409, "This conversation is finished. Start a new chat.")

        try:
            new = await agent_graph.ainvoke({**state, "transcript": text})
        except Exception:
            logger.exception("Graph failed on message")
            # Nothing is saved, so the user can safely retry.
            raise HTTPException(502, "The assistant is not available right now.")

        store.save_state(session_id, new)
        store.add_message(session_id, "user", text)

        reply = build_reply(new)
        _store_reply(session_id, reply)

    return reply