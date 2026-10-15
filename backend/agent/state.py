from typing import Any, Literal, TypedDict


class AgentState(TypedDict):
    """
    Every key here must be declared — LangGraph builds its state
    channels from this TypedDict, and any key a node returns that
    isn't listed gets silently dropped on the merge.
    """

    transcript: str

    # Keys are field "name"s from questions.json, so this is a
    # plain dict rather than a fixed TypedDict — the field set is
    # config-driven. A value can be interview_manager.NONE_VALUE,
    # meaning "candidate explicitly said they don't have this".
    candidate: dict[str, Any]

    current_question_index: int

    # The question object from questions.json currently being
    # asked. During "correcting" this is temporarily swapped for
    # whichever question owns the field being corrected — the index
    # itself is untouched, so the normal sequence resumes at the
    # summary afterward, not mid-list.
    current_question: dict | None

    # "interview"   — walking through questions.json in order
    # "summary"     — candidate is replying to the spoken review
    # "correcting"  — re-answering one field they asked to change
    # "finished"    — confirmed, interview over
    mode: Literal["interview", "summary", "correcting", "finished"]

    # Resolved list of segments to speak next — each item is either
    # {"kind": "static", "id": ..., "text": ...} or
    # {"kind": "value", "text": ...} (values already substituted in
    # by graph.py, so handler.py never touches state["candidate"]
    # directly). This is what handler.py's speak_segments() plays.
    current_prompt: list[dict]

    # Same shape as current_prompt, for an intent's own line
    # (e.g. "طيب قولي إيه اللي مش سامعه") or the summary
    # processing filler to be spoken before current_prompt.
    # Empty list when there's nothing extra to say.
    intent_prompt: list[dict]

    # How many times the current question has been retried in a
    # row (no_speech or wrong_answer). Reset to 0 on every accepted
    # answer or question change. Read by handler.py against
    # interview_manager.max_retries_per_question.
    retry_count: int

    extraction_success: bool

    # Why the last turn produced nothing: "no_speech" or
    # "wrong_answer". None on success, or when an intent was matched
    # instead (that's reported separately via intent_prompt).
    failure_reason: Literal["no_speech", "wrong_answer"] | None

    interview_finished: bool

    # True when the call is ending because of an intent like
    # call_later/not_interested, or because the candidate never
    # responded through the full silence escalation — NOT a normal
    # confirmed completion. handler.py uses this to skip the
    # completion clip (the intent's or silence prompt's own line
    # already said goodbye) but still save whatever was collected.
    ended_early: bool