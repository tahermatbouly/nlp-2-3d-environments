from typing import Any, Literal, TypedDict


class AgentState(TypedDict):
    """
    Every key here must be declared: LangGraph builds its state
    channels from this TypedDict, and any key a node returns that
    isn't listed gets silently dropped on the merge.
    """

    transcript: str

    # "en" or "ar_eg" - picks which text from the JSON is spoken.
    language: str

    # slot id -> value, for every slot from requirements_questions.json.
    # Types follow the slot type: int, float, bool, str, list[str],
    # or list[{"from", "to", "via"}] for room_connections.
    requirements: dict[str, Any]

    # Slot ids that are SETTLED: answered by the user, explicitly
    # declined ("you decide"), or filled from the default. A slot can
    # be resolved with value None (optional slot the user skipped).
    resolved: list[str]

    # Subset of `resolved` that was filled from a default rather than
    # stated by the user. Read back in the summary with a tag, and
    # saved in the final room graph as "assumed_fields".
    assumed: list[str]

    # "start"      - nothing said yet, next call plays the greeting
    # "gathering"  - asking for missing slots (or waiting for the
    #                user's first, open description)
    # "validating" - a validation rule failed and we asked about it
    # "summary"    - user is replying to the spoken summary
    # "correcting" - re-asking one slot the user wanted to change
    # "finished"   - confirmed (or ended early), conversation over
    mode: Literal[
        "start", "gathering", "validating", "summary", "correcting", "finished"
    ]

    # Slot ids asked in the last prompt (up to policy
    # max_questions_per_turn). Empty during the open first turn.
    current_slots: list[str]

    # Segments to speak next. Each item is
    # {"kind": "static", "id": ..., "text": ...}  (cacheable by id) or
    # {"kind": "live", "text": ...}                (synthesized fresh).
    current_prompt: list[dict]

    # Same shape, for an intent's own line ("Take your time.") to be
    # spoken before current_prompt. Empty when there is nothing extra.
    intent_prompt: list[dict]

    # Consecutive failed attempts on the current prompt.
    retry_count: int

    extraction_success: bool
    failure_reason: Literal["no_speech", "wrong_answer"] | None

    # The validation issue being asked about while mode == "validating":
    # {"rule": id, "kind": "warning" | "connect", "room": id | None}
    pending_validation: dict | None

    # Soft validation rules the user already confirmed ("yes, that's
    # fine"), so they are not asked again.
    acknowledged_rules: list[str]

    # Final output for Phase 2 (rooms + connections + assumptions).
    # None until the user confirms the summary.
    room_graph: dict | None

    conversation_finished: bool

    # True when the conversation ended through an intent (end_call) or
    # because a required slot could not be collected - NOT a normal
    # confirmed completion.
    ended_early: bool


def new_state(language: str = "ar_eg") -> AgentState:
    """The initial state handler code should pass to the first call."""

    return {
        "transcript": "",
        "language": language,
        "requirements": {},
        "resolved": [],
        "assumed": [],
        "mode": "start",
        "current_slots": [],
        "current_prompt": [],
        "intent_prompt": [],
        "retry_count": 0,
        "extraction_success": True,
        "failure_reason": None,
        "pending_validation": None,
        "acknowledged_rules": [],
        "room_graph": None,
        "conversation_finished": False,
        "ended_early": False,
    }