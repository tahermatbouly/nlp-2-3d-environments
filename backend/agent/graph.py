from langgraph.graph import StateGraph, START, END

from backend.agent.state import AgentState
from backend.agent.conversation_manager import ConversationManager
from backend.agent.extraction import (
    classify_confirmation_reply,
    is_obviously_invalid,
    parse_inline_correction_value,
    process_turn,
)


QUESTIONS_PATH = "Backend/agent/requirements_questions.json"

manager = ConversationManager(QUESTIONS_PATH)


# ============================================================
# Small helpers
# ============================================================

def _plain_text(segments: list[dict]) -> str:
    """What was actually said, used as CURRENT QUESTION context."""

    return " ".join(seg["text"] for seg in segments)


def _base(state: AgentState, requirements: dict, resolved: list, assumed: list, **changes) -> AgentState:
    """State after a successful turn: counters and flags reset."""

    return {
        **state,
        "requirements": requirements,
        "resolved": resolved,
        "assumed": assumed,
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
        "intent_prompt": [],
        **changes,
    }


def _summary_segments(state: AgentState) -> list[dict]:

    summary = manager.summary_text(
        state["requirements"], state["assumed"], state["language"]
    )

    return manager.confirmation_segments(summary, state["language"])


def _repeat_prompt(state: AgentState) -> list[dict]:
    """The right thing to say again for the current mode."""

    lang = state["language"]
    mode = state["mode"]

    if mode == "validating":
        return manager.validation_segments(
            state["pending_validation"], state["requirements"], lang
        )

    if mode == "summary":
        return _summary_segments(state)

    if state["current_slots"]:
        return manager.question_segments(
            state["current_slots"], state["requirements"], lang, repeat=True
        )

    return manager.message_segments("greeting", lang)


def _merge_edges(existing: list[dict], new: list[dict]) -> list[dict]:
    """New edges override older ones between the same two rooms."""

    merged = {frozenset((e["from"], e["to"])): e for e in existing}

    for edge in new:
        merged[frozenset((edge["from"], edge["to"]))] = edge

    return list(merged.values())


def _apply_result(requirements: dict, resolved: list, assumed: list, result: dict) -> set[str]:
    """Write an extraction result into the requirements. Returns the
    slot ids that were actually settled by it."""

    touched: set[str] = set()

    for sid, value in result.get("updates", {}).items():

        slot = manager.get_slot(sid)

        if slot is not None and slot["type"] == "edge_list":
            value = _merge_edges(requirements.get(sid) or [], value)

        requirements[sid] = value

        if sid not in resolved:
            resolved.append(sid)

        if sid in assumed:
            assumed.remove(sid)

        touched.add(sid)

    for sid in result.get("use_default_for", []):

        if manager.settle_with_default(sid, requirements, resolved, assumed):
            touched.add(sid)

        elif not manager.get_slot(sid).get("required"):
            # "I don't care" about an optional slot with no default:
            # settled with no value, never asked again.
            if sid not in resolved:
                resolved.append(sid)

            touched.add(sid)

    manager.reconcile(requirements, resolved, assumed)

    return touched


# ============================================================
# Start
# ============================================================

async def start_conversation(state: AgentState) -> AgentState:
    """Play the greeting and wait for the user's open description."""

    print("[GRAPH] Starting conversation")

    lang = state.get("language") or manager.default_language

    return {
        **state,
        "language": lang,
        "requirements": {},
        "resolved": [],
        "assumed": [],
        "acknowledged_rules": [],
        "mode": "gathering",
        "current_slots": [],
        "current_prompt": manager.message_segments("greeting", lang),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "pending_validation": None,
        "room_graph": None,
        "conversation_finished": False,
        "ended_early": False,
        "transcript": "",
    }


# ============================================================
# Choosing what happens next
# ============================================================

def _next_step(
    state: AgentState,
    requirements: dict,
    resolved: list,
    assumed: list,
    intent_prompt: list | None = None,
) -> AgentState:
    """
    1. Missing slots left        -> ask them (max N per turn)
    2. Fill silent defaults, then run validation rules
       - rule failed             -> ask about it
    3. Everything fine           -> spoken summary
    """

    lang = state["language"]
    base = _base(state, requirements, resolved, assumed)

    ids = manager.next_slots(requirements, resolved)

    if ids:

        print("[GRAPH] Asking:", ids)

        result = {
            **base,
            "mode": "gathering",
            "current_slots": ids,
            "pending_validation": None,
            "current_prompt": manager.question_segments(ids, requirements, lang),
        }

    else:

        manager.fill_defaults(requirements, resolved, assumed)

        issue = manager.validate(requirements, state["acknowledged_rules"])

        if issue and issue["kind"] == "reask":

            requirements.pop(issue["slot"], None)

            if issue["slot"] in resolved:
                resolved.remove(issue["slot"])

            return _next_step(state, requirements, resolved, assumed, intent_prompt)

        if issue:

            print("[GRAPH] Validation issue:", issue)

            result = {
                **base,
                "mode": "validating",
                "current_slots": [],
                "pending_validation": issue,
                "current_prompt": manager.validation_segments(issue, requirements, lang),
            }

        else:

            print("[GRAPH] Everything collected -> summary")

            result = _build_summary(base)

    if intent_prompt:
        result["intent_prompt"] = intent_prompt

    return result


def _build_summary(state: AgentState) -> AgentState:

    return {
        **state,
        "mode": "summary",
        "current_slots": [],
        "pending_validation": None,
        "current_prompt": _summary_segments(state),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
    }


def _cannot_continue(state: AgentState) -> AgentState:

    print("[GRAPH] Cannot continue: required slot never answered")

    return {
        **state,
        "mode": "finished",
        "current_slots": [],
        "current_prompt": manager.message_segments("cannot_continue", state["language"]),
        "intent_prompt": [],
        "conversation_finished": True,
        "ended_early": True,
        "extraction_success": False,
        "transcript": "",
    }


def _retry(state: AgentState, reason: str) -> AgentState:
    """Say it again, or give up on this prompt after max_reprompts."""

    retry_count = state["retry_count"] + 1

    if retry_count > manager.max_reprompts:
        return _give_up(state)

    return {
        **state,
        "current_prompt": _repeat_prompt(state),
        "intent_prompt": [],
        "retry_count": retry_count,
        "extraction_success": False,
        "failure_reason": reason,
        "transcript": "",
    }


def _give_up(state: AgentState) -> AgentState:
    """
    Too many failed attempts on one prompt:
      - correcting   -> keep the old value, back to the summary
      - validating   -> fix it automatically / accept the warning
      - gathering    -> use the default; optional slots are skipped;
                        a required slot with no default ends the call
                        (an open first turn just starts the questions)
    """

    requirements = dict(state["requirements"])
    resolved = list(state["resolved"])
    assumed = list(state["assumed"])

    mode = state["mode"]

    if mode == "correcting":
        return _build_summary(_base(state, requirements, resolved, assumed))

    if mode == "validating":

        pending = state["pending_validation"]
        acknowledged = list(state["acknowledged_rules"])

        if pending["kind"] == "connect":
            manager.auto_connect(requirements, pending["room"])
        else:
            acknowledged.append(pending["rule"])

        return _next_step(
            {**state, "acknowledged_rules": acknowledged},
            requirements,
            resolved,
            assumed,
        )

    for sid in state["current_slots"]:

        if sid in resolved:
            continue

        if manager.settle_with_default(sid, requirements, resolved, assumed):
            continue

        if manager.get_slot(sid).get("required"):
            return _cannot_continue(state)

        resolved.append(sid)

    return _next_step(state, requirements, resolved, assumed)


# ============================================================
# Gathering (and correcting one slot)
# ============================================================

async def process_answer(state: AgentState) -> AgentState:

    transcript = state["transcript"].strip()
    lang = state["language"]

    if is_obviously_invalid(transcript):

        print("[GRAPH] Obviously invalid transcript")

        return _retry(state, "no_speech")

    requirements = dict(state["requirements"])
    resolved = list(state["resolved"])
    assumed = list(state["assumed"])

    current_ids = state["current_slots"]
    correcting = state["mode"] == "correcting"

    current_slots = [manager.get_slot(i) for i in current_ids]

    slots = (
        current_slots
        if correcting
        else manager.extractable_slots(requirements, resolved, current_ids)
    )

    result = await process_turn(
        question_text=_plain_text(state["current_prompt"]),
        slots=slots,
        current_slots=current_slots,
        intents=manager.intents,
        transcript=transcript,
        rooms_context=manager.rooms_context(requirements, resolved, lang),
        room_ids=manager.room_ids(requirements, resolved),
    )

    print("[GRAPH] Turn result:", result)

    if result["kind"] == "intent":
        return await _handle_intent(state, result["intent_id"])

    if result["kind"] != "answer":
        return _retry(state, "wrong_answer")

    touched = _apply_result(requirements, resolved, assumed, result)

    # Volunteered extras are kept, but the question that was asked
    # must be answered before we move on.
    if current_ids and not touched & set(current_ids):

        print("[GRAPH] Current question not answered:", current_ids, "| got:", sorted(touched))

        return _retry(
            {**state, "requirements": requirements, "resolved": resolved, "assumed": assumed},
            "wrong_answer",
        )

    if correcting:
        state = {**state, "acknowledged_rules": []}

    return _next_step(state, requirements, resolved, assumed)


# ============================================================
# Intents
# ============================================================

async def _handle_intent(state: AgentState, intent_id: str) -> AgentState:

    intent = manager.get_intent(intent_id)

    if intent is None:
        return _retry(state, "wrong_answer")

    line = manager.intent_segments(intent, state["language"])
    action = intent.get("then", "repeat_question")

    print("[GRAPH] Intent:", intent_id, "| then:", action)

    quiet = {
        "extraction_success": False,
        "failure_reason": None,
        "transcript": "",
    }

    # Say the line, keep the same prompt active, no repeat.
    if action == "wait":
        return {**state, "current_prompt": [], "intent_prompt": line, **quiet}

    # Say the line, then repeat. Not a failed attempt, so
    # retry_count is untouched.
    if action == "repeat_question":
        return {
            **state,
            "current_prompt": _repeat_prompt(state),
            "intent_prompt": line,
            **quiet,
        }

    # The line is already a goodbye. Stop here.
    if action == "end_call":
        return {
            **state,
            "mode": "finished",
            "current_slots": [],
            "current_prompt": [],
            "intent_prompt": line,
            "conversation_finished": True,
            "ended_early": True,
            "extraction_success": True,
            "failure_reason": None,
            "transcript": "",
        }

    return _retry(state, "wrong_answer")


# ============================================================
# Corrections (shared by the summary and soft warnings)
# ============================================================

def _apply_correction(
    state: AgentState,
    requirements: dict,
    resolved: list,
    assumed: list,
    decision: dict,
) -> AgentState:
    """
    The user wants to change one field. If they also said the new
    value, apply it straight away. Otherwise re-ask that slot.
    """

    lang = state["language"]
    field = decision["field"]
    slot = manager.get_slot(field)

    # Values changed, so earlier "that's fine" answers are stale.
    state = {**state, "acknowledged_rules": []}

    if slot is None:
        return _next_step(state, requirements, resolved, assumed)

    new_value = decision.get("new_value")

    if new_value:

        parsed = parse_inline_correction_value(
            manager, slot, new_value, manager.room_ids(requirements, resolved)
        )

        if parsed is not None:

            print("[GRAPH] Inline correction:", field, "=", parsed)

            requirements[field] = parsed

            if field not in resolved:
                resolved.append(field)

            if field in assumed:
                assumed.remove(field)

            manager.reconcile(requirements, resolved, assumed)

            return _next_step(state, requirements, resolved, assumed)

    # Changing connections only makes sense in manual mode.
    if field == "room_connections":

        requirements["connections_mode"] = "manual"

        if "connections_mode" not in resolved:
            resolved.append("connections_mode")

    print("[GRAPH] Correcting:", field)

    return _base(
        state,
        requirements,
        resolved,
        assumed,
        mode="correcting",
        current_slots=[field],
        pending_validation=None,
        current_prompt=manager.question_segments([field], requirements, lang),
    )


# ============================================================
# Validation reply
# ============================================================

async def process_validation_reply(state: AgentState) -> AgentState:

    transcript = state["transcript"].strip()
    lang = state["language"]

    if is_obviously_invalid(transcript):
        return _retry(state, "no_speech")

    pending = state["pending_validation"]

    requirements = dict(state["requirements"])
    resolved = list(state["resolved"])
    assumed = list(state["assumed"])

    # --------------------------------------------------------
    # A room is isolated / unreachable: which room does it open to?
    # --------------------------------------------------------

    if pending["kind"] == "connect":

        repair = manager.repair_slot()

        result = await process_turn(
            question_text=_plain_text(state["current_prompt"]),
            slots=[repair],
            current_slots=[repair],
            intents=manager.intents,
            transcript=transcript,
            rooms_context=manager.rooms_context(requirements, resolved, lang),
            room_ids=manager.room_ids(requirements, resolved),
        )

        print("[GRAPH] Repair result:", result)

        if result["kind"] == "intent":
            return await _handle_intent(state, result["intent_id"])

        target = result.get("updates", {}).get("repair_target") if result["kind"] == "answer" else None

        if not target or target == pending["room"]:
            return _retry(state, "wrong_answer")

        requirements["room_connections"] = list(requirements.get("room_connections") or []) + [
            {"from": pending["room"], "to": target, "via": "door"}
        ]

        return _next_step(state, requirements, resolved, assumed)

    # --------------------------------------------------------
    # Soft warning: keep as is, or change something?
    # --------------------------------------------------------

    decision = await classify_confirmation_reply(
        transcript,
        _plain_text(state["current_prompt"]),
        manager.correctable_slots(),
    )

    print("[GRAPH] Warning reply:", decision)

    if decision["action"] == "confirm":

        return _next_step(
            {**state, "acknowledged_rules": state["acknowledged_rules"] + [pending["rule"]]},
            requirements,
            resolved,
            assumed,
        )

    if decision["action"] == "correct":
        return _apply_correction(state, requirements, resolved, assumed, decision)

    return _retry(state, "wrong_answer")


# ============================================================
# Summary reply
# ============================================================

async def process_summary_reply(state: AgentState) -> AgentState:

    transcript = state["transcript"].strip()
    lang = state["language"]

    replay = {
        **state,
        "current_prompt": _summary_segments(state),
        "intent_prompt": [],
        "extraction_success": False,
        "failure_reason": None,
        "transcript": "",
    }

    if is_obviously_invalid(transcript):

        print("[GRAPH] Summary reply obviously invalid, replaying")

        return replay

    decision = await classify_confirmation_reply(
        transcript,
        _plain_text(state["current_prompt"]),
        manager.correctable_slots(),
    )

    print("[GRAPH] Summary reply:", decision)

    requirements = dict(state["requirements"])
    resolved = list(state["resolved"])
    assumed = list(state["assumed"])

    if decision["action"] == "confirm":

        return {
            **state,
            "mode": "finished",
            "current_slots": [],
            "current_prompt": manager.message_segments("completion", lang),
            "intent_prompt": [],
            "room_graph": manager.build_room_graph(requirements, assumed),
            "conversation_finished": True,
            "ended_early": False,
            "extraction_success": True,
            "failure_reason": None,
            "transcript": "",
        }

    if decision["action"] == "correct":
        return _apply_correction(state, requirements, resolved, assumed, decision)

    print("[GRAPH] Summary reply unclear, replaying summary")

    return replay


# ============================================================
# Routing
# ============================================================

def route_from_start(state: AgentState):
    """
    The mode decides everything. "start" is only the state handler
    code creates with new_state(): the first call plays the greeting,
    and every later call carries a transcript. An empty transcript
    mid-conversation is just a no_speech retry, it never restarts.
    """

    return {
        "start": "start",
        "gathering": "answer",
        "correcting": "answer",
        "validating": "validation",
        "summary": "summary",
    }.get(state.get("mode", "start"), "done")


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("start_conversation", start_conversation)
builder.add_node("process_answer", process_answer)
builder.add_node("process_validation_reply", process_validation_reply)
builder.add_node("process_summary_reply", process_summary_reply)

builder.add_conditional_edges(
    START,
    route_from_start,
    {
        "start": "start_conversation",
        "answer": "process_answer",
        "validation": "process_validation_reply",
        "summary": "process_summary_reply",
        "done": END,
    },
)

builder.add_edge("start_conversation", END)
builder.add_edge("process_answer", END)
builder.add_edge("process_validation_reply", END)
builder.add_edge("process_summary_reply", END)

agent_graph = builder.compile()