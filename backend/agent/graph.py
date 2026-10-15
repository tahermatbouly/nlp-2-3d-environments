from langgraph.graph import StateGraph, START, END

from Backend.agent.state import AgentState
from Backend.agent.interview_manager import InterviewManager, NONE_VALUE
from Backend.agent.extraction import (
    build_summary_segments,
    classify_summary_reply,
    is_obviously_invalid,
    parse_inline_correction_value,
    process_turn,
)

import re


QUESTIONS_PATH = "Backend/agent/questions.json"

interview_manager = InterviewManager(QUESTIONS_PATH)


# ============================================================
# Segment resolution
# ============================================================
# Turns a question's raw segment list (which may contain "value"
# segments referencing a field) into what handler.py actually
# speaks — plain {"kind": "static"/"value"/"live", ...} dicts with
# every value already substituted from the candidate dict. Keeping
# this resolution in the graph means handler.py never touches
# state["candidate"] directly.

def _format(value) -> str:

    if isinstance(value, list):
        return "، ".join(str(item) for item in value)

    return str(value)


def _first_name(value) -> str:
    """
    Spoken form for recalling a person: first token only, so
    "طاهر محمد علي" becomes "طاهر".
    """

    text = _format(value).strip()

    if not text:
        return text

    return text.split()[0]


def _spoken_field_value(field: str, value) -> str:
    """How a field's value should sound when spoken back."""

    if field == "candidate_name":
        return _first_name(value)

    return _format(value)


def _resolve_live_template(
    template: str,
    candidate: dict,
) -> str | None:
    """
    Substitute {field} placeholders. Returns None if any required
    field is missing so the segment is skipped rather than spoken
    with a hole.
    """

    fields = re.findall(r"\{(\w+)\}", template)
    text = template

    for field in fields:

        value = candidate.get(field)

        if value is None or value == NONE_VALUE:
            return None

        text = text.replace(
            "{" + field + "}",
            _spoken_field_value(field, value),
        )

    return text


def _resolve_segments(
    segments: list[dict],
    candidate: dict,
) -> list[dict]:

    resolved = []

    for seg in segments:

        if seg["type"] == "static":

            resolved.append(
                {"kind": "static", "id": seg["id"], "text": seg["text"]}
            )

        elif seg["type"] == "live":

            # Whole phrase synthesized fresh each time (no TTS cache).
            text = _resolve_live_template(seg["template"], candidate)

            if text is None:
                continue

            resolved.append({"kind": "live", "text": text})

        elif seg["type"] == "value":

            value = candidate.get(seg["field"])

            if value is None or value == NONE_VALUE:
                continue

            resolved.append(
                {
                    "kind": "value",
                    "text": _spoken_field_value(seg["field"], value),
                }
            )

        elif seg["type"] == "value_text":

            # Already-formatted text (used by build_summary_segments,
            # which resolves the value itself before this point).
            resolved.append({"kind": "value", "text": seg["text"]})

    return resolved


def _plain_text(segments: list[dict], candidate: dict) -> str:
    """
    The full sentence actually spoken for these segments, used as
    the CURRENT QUESTION context in the extraction prompt. Built by
    resolving and joining every segment (not just assuming the last
    one is static) so this stays correct regardless of where a
    "value" segment falls in the sentence.
    """

    resolved = _resolve_segments(segments, candidate)

    return " ".join(seg["text"] for seg in resolved)


def _question_prompt(
    question: dict,
    candidate: dict,
    repeat: bool = False,
) -> list[dict]:

    segments = (
        question.get("repeat_segments") or question["segments"]
        if repeat
        else question["segments"]
    )

    return _resolve_segments(segments, candidate)


# ============================================================
# Start
# ============================================================

async def start_interview(state: AgentState) -> AgentState:

    print("[GRAPH] Starting interview")

    first_question = interview_manager.get_question(0)

    return {
        **state,
        "candidate": {},
        "current_question_index": 0,
        "current_question": first_question,
        "mode": "interview",
        "current_prompt": _question_prompt(first_question, {}),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "interview_finished": False,
        "ended_early": False,
        "transcript": "",
    }


# ============================================================
# Advance / summary helpers
# ============================================================

def _field_is_filled(candidate: dict, field_name: str) -> bool:
    """True once we have a stored value, including explicit NONE."""

    return (
        field_name in candidate
        and candidate[field_name] is not None
    )


def _question_is_satisfied(
    question: dict,
    candidate: dict,
) -> bool:
    """
    A question is done when every field it asks for is already
    filled — used to skip ahead when the candidate volunteered
    answers earlier.
    """

    targets = interview_manager.target_fields(question)

    if not targets:
        return False

    return all(
        _field_is_filled(candidate, name) for name in targets
    )


def _extractable_fields(
    question: dict,
    candidate: dict,
    *,
    correcting: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (current_fields, all_fields_for_tool).

    In correcting mode only the fields for that question are
    exposed. In interview mode the tool also sees every still-empty
    interview field so volunteered future answers can be stored.
    """

    target_names = interview_manager.target_fields(question)

    current_fields = [
        f
        for name in target_names
        if (f := interview_manager.get_field(name)) is not None
    ]

    if correcting:
        return current_fields, current_fields

    seen = {f["name"] for f in current_fields}
    extra = []

    for field in interview_manager.fields:

        name = field["name"]

        if name in seen:
            continue

        if _field_is_filled(candidate, name):
            continue

        # Only pull in fields that belong to the interview sequence
        # (or the free-form notes catch-all).
        if field.get("asked_by") or name == "notes":
            extra.append(field)
            seen.add(name)

    return current_fields, current_fields + extra


def _advance_or_summarize(state: AgentState) -> AgentState:

    candidate = state["candidate"]

    next_index = state["current_question_index"] + 1

    while True:

        next_question = interview_manager.get_question(next_index)

        if next_question is None:

            print("[GRAPH] Out of questions -> summary")

            return _build_summary(state)

        if _question_is_satisfied(next_question, candidate):

            print(
                "[GRAPH] Skipping already-filled question:",
                next_question["id"],
            )

            next_index += 1
            continue

        break

    print("[GRAPH] Advancing to:", next_question["id"])

    return {
        **state,
        "current_question_index": next_index,
        "current_question": next_question,
        "mode": "interview",
        "current_prompt": _question_prompt(next_question, candidate),
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
    }


def _build_summary(state: AgentState) -> AgentState:

    segments = build_summary_segments(
        state["candidate"], interview_manager
    )

    candidate = state["candidate"]

    # Speak the short "one moment" filler first so the candidate
    # hears something while summary value clips are still synthesizing.
    # Only on entry into summary (not when rebuilding after an
    # inline correction) — they already heard it once.
    if state.get("mode") == "summary":
        intent_prompt: list[dict] = []
    else:
        processing = interview_manager.summary_config.get(
            "processing", []
        )
        intent_prompt = _resolve_segments(processing, candidate)

    return {
        **state,
        "current_question": None,
        "mode": "summary",
        "current_prompt": _resolve_segments(segments, candidate),
        "intent_prompt": intent_prompt,
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "transcript": "",
    }


def _retry_or_skip(state: AgentState, failure_reason: str) -> AgentState:
    """
    Repeats the current question, unless it has already failed
    max_retries_per_question times in a row — then it moves on
    rather than looping forever. Required fields still get skipped
    past this limit (better an incomplete interview than a stuck
    call); skip_optional_after_max_retries only changes whether an
    optional field's absence is worth distinguishing in logs, since
    either way we must advance.
    """

    question = state["current_question"]
    candidate = state["candidate"]

    retry_count = state["retry_count"] + 1

    if retry_count >= interview_manager.max_retries_per_question:

        print(
            "[GRAPH] Max retries reached for",
            question["id"] if question else None,
            "- skipping",
        )

        return _advance_or_summarize({**state, "retry_count": 0})

    return {
        **state,
        "current_prompt": _question_prompt(
            question, candidate, repeat=True
        ),
        "intent_prompt": [],
        "retry_count": retry_count,
        "extraction_success": False,
        "failure_reason": failure_reason,
        "transcript": "",
    }


# ============================================================
# Process an answer to a normal (or correcting) question
# ============================================================

async def process_answer(state: AgentState) -> AgentState:

    question = state["current_question"]
    transcript = state["transcript"].strip()
    candidate = dict(state["candidate"])

    # ----------------------------------------------------
    # Obvious silence/filler — skip the LLM call entirely.
    # ----------------------------------------------------

    if is_obviously_invalid(transcript):

        print("[GRAPH] Obviously invalid transcript")

        return _retry_or_skip(state, "no_speech")

    target_field_names = interview_manager.target_fields(question)

    current_fields, extractable_fields = _extractable_fields(
        question,
        candidate,
        correcting=state["mode"] == "correcting",
    )

    result = await process_turn(
        question_text=_plain_text(question["segments"], candidate),
        fields=extractable_fields,
        current_fields=current_fields,
        intents=interview_manager.intents,
        transcript=transcript,
    )

    print("[GRAPH] Turn result:", result)

    # ----------------------------------------------------
    # Intent matched — not an answer at all.
    # ----------------------------------------------------

    if result["kind"] == "intent":

        return await _handle_intent(state, result["intent_id"])

    # ----------------------------------------------------
    # No usable answer, no intent match.
    # ----------------------------------------------------

    if result["kind"] != "answer" or not result["updates"]:

        return _retry_or_skip(state, "wrong_answer")

    # ----------------------------------------------------
    # Real answer — may include future fields volunteered early.
    # ----------------------------------------------------

    candidate.update(result["updates"])

    # The current question still must be addressed. Extra fields
    # alone are not enough to move on.
    if not any(
        _field_is_filled(candidate, name)
        for name in target_field_names
    ):

        print(
            "[GRAPH] Updates missed current question fields:",
            target_field_names,
            "| got:",
            list(result["updates"]),
        )

        return _retry_or_skip(
            {**state, "candidate": candidate},
            "wrong_answer",
        )

    new_state = {**state, "candidate": candidate}

    if state["mode"] == "correcting":

        print("[GRAPH] Correction applied, rebuilding summary")

        return _build_summary(new_state)

    return _advance_or_summarize(new_state)


# ============================================================
# Intents
# ============================================================

async def _handle_intent(state: AgentState, intent_id: str) -> AgentState:

    intent = interview_manager.get_intent(intent_id)

    if intent is None:
        # Unknown id somehow — fall back to a normal retry rather
        # than crash.
        return _retry_or_skip(state, "wrong_answer")

    intent_prompt = [
        {"kind": "static", "id": intent["id"], "text": intent["text"]}
    ]

    action = intent.get("then", "repeat_question")

    question = state["current_question"]
    candidate = state["candidate"]

    print("[GRAPH] Intent:", intent_id, "| then:", action)

    # --------------------------------------------------------
    # wait: say the line, keep the exact same question active,
    # don't touch retry_count (asking for a moment isn't a failed
    # answer) and don't repeat the question — silence afterward is
    # just them thinking.
    # --------------------------------------------------------

    if action == "wait":

        return {
            **state,
            "current_prompt": [],
            "intent_prompt": intent_prompt,
            "extraction_success": False,
            "failure_reason": None,
            "transcript": "",
        }

    # --------------------------------------------------------
    # repeat_question: say the line, then repeat the question —
    # doesn't count against the retry limit, this isn't a failed
    # extraction attempt.
    # --------------------------------------------------------

    if action == "repeat_question":

        return {
            **state,
            "current_prompt": _question_prompt(
                question, candidate, repeat=True
            ),
            "intent_prompt": intent_prompt,
            "extraction_success": False,
            "failure_reason": None,
            "transcript": "",
        }

    # --------------------------------------------------------
    # skip_question: say the line, move on without filling this
    # question's fields.
    # --------------------------------------------------------

    if action == "skip_question":

        advanced = _advance_or_summarize({**state, "retry_count": 0})

        # If skip lands on summary, keep the processing filler so
        # value-clip TTS still overlaps speech (skip line alone is
        # usually too short to cover synthesis).
        if advanced.get("mode") == "summary":
            summary_filler = advanced.get("intent_prompt") or []
        else:
            summary_filler = []

        return {
            **advanced,
            "intent_prompt": intent_prompt + summary_filler,
        }

    # --------------------------------------------------------
    # end_call: say the line (it's already a full goodbye), stop
    # here. handler.py sees ended_early and saves without playing
    # the normal completion clip.
    # --------------------------------------------------------

    if action == "end_call":

        return {
            **state,
            "current_prompt": [],
            "intent_prompt": intent_prompt,
            "mode": "finished",
            "extraction_success": True,
            "failure_reason": None,
            "interview_finished": True,
            "ended_early": True,
            "transcript": "",
        }

    return _retry_or_skip(state, "wrong_answer")


# ============================================================
# Summary
# ============================================================

async def process_summary_reply(state: AgentState) -> AgentState:

    transcript = state["transcript"].strip()

    if is_obviously_invalid(transcript):

        print("[GRAPH] Summary reply obviously invalid, replaying")

        return {
            **state,
            "current_prompt": _resolve_segments(
                build_summary_segments(
                    state["candidate"], interview_manager
                ),
                state["candidate"],
            ),
            "intent_prompt": [],
            "extraction_success": False,
            "failure_reason": None,
            "transcript": "",
        }

    decision = await classify_summary_reply(
        transcript, interview_manager.fields
    )

    print("[GRAPH] Summary reply:", decision)

    if decision["action"] == "confirm":

        candidate = state["candidate"]

        return {
            **state,
            "mode": "finished",
            "current_question": None,
            "current_prompt": _resolve_segments(
                interview_manager.completion_segments, candidate
            ),
            "intent_prompt": [],
            "extraction_success": True,
            "failure_reason": None,
            "interview_finished": True,
            "ended_early": False,
            "transcript": "",
        }

    if decision["action"] == "correct":

        field_name = decision["field"]
        inline_value = decision.get("new_value")

        # ----------------------------------------------------
        # Inline correction: they said what to change AND the
        # new value in one utterance — apply it and rebuild the
        # summary without re-asking the field question.
        # ----------------------------------------------------

        if inline_value and field_name:

            field_def = interview_manager.get_field(field_name)

            if field_def is not None:

                parsed = parse_inline_correction_value(
                    field_def, inline_value
                )

                if parsed is not None:

                    candidate = dict(state["candidate"])
                    candidate[field_name] = parsed

                    print(
                        "[GRAPH] Inline correction applied:",
                        field_name,
                        "=",
                        parsed,
                    )

                    return _build_summary(
                        {**state, "candidate": candidate}
                    )

        target_question = interview_manager.question_for_field(
            field_name
        )

        if target_question is not None:

            print("[GRAPH] Correcting:", target_question["id"])

            return {
                **state,
                "mode": "correcting",
                "current_question": target_question,
                "current_prompt": _question_prompt(
                    target_question, state["candidate"], repeat=True
                ),
                "intent_prompt": [],
                "retry_count": 0,
                "extraction_success": True,
                "failure_reason": None,
                "transcript": "",
            }

    print("[GRAPH] Summary reply unclear, replaying summary")

    return {
        **state,
        "current_prompt": _resolve_segments(
            build_summary_segments(
                state["candidate"], interview_manager
            ),
            state["candidate"],
        ),
        "intent_prompt": [],
        "extraction_success": False,
        "failure_reason": None,
        "transcript": "",
    }


# ============================================================
# Routing
# ============================================================

def route_from_start(state: AgentState):
    """
    current_question_index starts at -1 in handler.py's initial
    state dict and is only ever set to >= 0 by start_interview.
    That — NOT "is the transcript empty" — is what actually
    distinguishes "this is the very first call" from "this is a
    normal turn where STT happened to come back empty".

    An empty transcript mid-interview must NOT re-trigger
    start_interview: that would silently wipe state["candidate"]
    and restart the whole call from question 0. process_answer's
    and process_summary_reply's own is_obviously_invalid() checks
    already treat "" correctly as a no_speech retry — there's
    nothing left for this router to special-case.
    """

    if state.get("current_question_index", -1) < 0:
        return "start"

    mode = state.get("mode", "interview")

    if mode == "summary":
        return "summary"

    return "answer"


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("start_interview", start_interview)
builder.add_node("process_answer", process_answer)
builder.add_node("process_summary_reply", process_summary_reply)

builder.add_conditional_edges(
    START,
    route_from_start,
    {
        "start": "start_interview",
        "answer": "process_answer",
        "summary": "process_summary_reply",
    },
)

builder.add_edge("start_interview", END)
builder.add_edge("process_answer", END)
builder.add_edge("process_summary_reply", END)

agent_graph = builder.compile()