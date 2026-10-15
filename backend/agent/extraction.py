from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from Backend import config
from Backend.agent.interview_manager import NONE_VALUE


# ============================================================
# LLM
# ============================================================
# temperature=0: this is extraction/classification, not writing.
# Any randomness here shows up as flaky tool-call behaviour on
# borderline answers.

llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
)


# ============================================================
# Known STT artifacts / obviously empty answers
# ============================================================
# Only treat true empties and known STT garbage as invalid here.
# Short conversational words (تمام، نعم، اه، لا، مش عارف) can be
# real answers or summary confirmations — let the LLM decide.

INVALID_TRANSCRIPTS = {
    "",
    "@@فراغ",
    "فراغ",
}


def is_obviously_invalid(transcript: str) -> bool:

    normalized = transcript.strip().lower()

    return normalized in INVALID_TRANSCRIPTS


# ============================================================
# Field update tool
# ============================================================

def _build_update_tool(fields: list[dict]) -> StructuredTool:

    model_fields = {}

    for field in fields:

        annotation = (
            list[str] | None if field.get("multi") else str | None
        )

        model_fields[field["name"]] = (
            annotation,
            Field(
                default=None,
                description=(
                    f"{field['description']} Set to the exact "
                    "string \"NONE\" (or [\"NONE\"] for a list "
                    "field) if the candidate explicitly said they "
                    "don't have this — that is different from not "
                    "mentioning it at all."
                ),
            ),
        )

    UpdateModel = create_model("AnswerUpdate", **model_fields)

    def update_candidate_info(**kwargs):
        return kwargs

    return StructuredTool.from_function(
        func=update_candidate_info,
        name="update_candidate_info",
        description=(
            "Call this when the candidate answered the CURRENT "
            "QUESTION with real information (or explicitly said "
            "they don't have it). Also fill any OTHER listed "
            "fields they clearly volunteered in the same "
            "utterance. Never call this for greetings or "
            "meta-conversation alone."
        ),
        args_schema=UpdateModel,
    )


def _build_intent_tool(intents: list[dict]) -> StructuredTool | None:

    if not intents:
        return None

    from typing import Literal

    ids = tuple(i["id"] for i in intents)

    IntentModel = create_model(
        "IntentMatch",
        intent_id=(
            Literal[ids],
            Field(..., description="Which situation this is."),
        ),
    )

    def match_intent(intent_id: str):
        return {"intent_id": intent_id}

    return StructuredTool.from_function(
        func=match_intent,
        name="match_intent",
        description=(
            "Call this when the candidate said something that is "
            "NOT an answer to the current question, but matches "
            "one of the known situations below (e.g. they can't "
            "hear, ask you to repeat, ask who's calling, ask to be "
            "called later). Do not call this for a normal answer."
        ),
        args_schema=IntentModel,
    )


def _describe_fields(fields: list[dict]) -> str:

    return "\n".join(
        f"- {f['name']} ({'list of strings' if f.get('multi') else 'string'}): {f['description']}"
        for f in fields
    )


def _describe_intents(intents: list[dict]) -> str:

    lines = []

    for intent in intents:

        examples = " | ".join(intent.get("examples", []))

        lines.append(
            f"- {intent['id']}: {intent['description']}\n"
            f"  examples: {examples}"
        )

    return "\n".join(lines)


TURN_PROMPT = """
You are handling one turn of an Egyptian Arabic recruitment phone
interview.

CURRENT QUESTION (what was just asked — this must be answered
unless the candidate clearly already covered it in this utterance):

{question}

CURRENT QUESTION FIELDS (at least one of these must be filled if
the candidate is answering normally):

{current_fields}

Decide exactly ONE of three things:

1. The candidate ANSWERED the current question (or explicitly said
   they don't have the thing being asked about). They may ALSO
   volunteer answers to later questions in the same utterance
   (e.g. name + years of experience together)
   -> call update_candidate_info with EVERY field they clearly
      stated — current and future. Leave unmentioned fields null.

2. The candidate said something else entirely — they can't hear,
   ask you to repeat, ask who's calling, ask to be called later,
   etc. — matching one of the situations below
   -> call match_intent

3. Neither of the above clearly applies
   -> call NOTHING

Never call both tools. Never guess a field value you are not
confident about. Never invent values for fields they did not
mention.

ALL FIELDS YOU MAY FILL THIS TURN (current + still-empty later
ones):

{fields}

KNOWN SITUATIONS (for match_intent):

{intents}

RULES:

- "لسه متخرج" or "ممعنديش خبرة" for years_of_experience means "0".
  Zero experience is a real answer, not a missing one.
- For list fields, put each skill/tool as its own separate item.
- The speech comes from an imperfect speech-to-text system; extract
  what is clearly there and ignore obvious noise.
- Only match an intent if the candidate's words clearly fit one of
  the situations above — a real (even partial) answer to the
  question always takes priority over an intent match.
- Extra volunteered info is a bonus: still fill the current
  question field(s) when they answered those too.

CANDIDATE SAID:

{transcript}
"""


async def process_turn(
    question_text: str,
    fields: list[dict],
    intents: list[dict],
    transcript: str,
    current_fields: list[dict] | None = None,
) -> dict:
    """
    One LLM call that decides between three outcomes, so a normal
    turn never costs more than a single round trip:

        {"kind": "answer", "updates": {...}}
        {"kind": "intent", "intent_id": "..."}
        {"kind": "none"}

    `fields` may include later interview fields so volunteered
    answers can be captured early. `current_fields` is what the
    prompt highlights as required for this turn.
    """

    current_fields = current_fields if current_fields is not None else fields

    update_tool = _build_update_tool(fields)
    intent_tool = _build_intent_tool(intents)

    tools = [update_tool] + ([intent_tool] if intent_tool else [])

    prompt = TURN_PROMPT.format(
        question=question_text,
        current_fields=_describe_fields(current_fields),
        fields=_describe_fields(fields),
        intents=_describe_intents(intents),
        transcript=transcript,
    )

    response = await llm.bind_tools(tools).ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=transcript),
        ]
    )

    print("[EXTRACTION] Tool calls:", response.tool_calls)

    if not response.tool_calls:
        return {"kind": "none"}

    for tool_call in response.tool_calls:

        if tool_call["name"] == "match_intent":

            intent_id = tool_call.get("args", {}).get("intent_id")

            if intent_id:
                return {"kind": "intent", "intent_id": intent_id}

        if tool_call["name"] == "update_candidate_info":

            updates = _clean_updates(
                tool_call.get("args", {}), fields
            )

            if updates:
                return {"kind": "answer", "updates": updates}

    return {"kind": "none"}


def _clean_updates(args: dict, fields: list[dict]) -> dict:

    allowed = {f["name"] for f in fields}

    updates = {}

    for name, value in args.items():

        if name not in allowed or value is None:
            continue

        if isinstance(value, str):

            value = value.strip()

            if not value:
                continue

            if value.upper() == "NONE":
                value = NONE_VALUE

        elif isinstance(value, list):

            cleaned = [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]

            if not cleaned:
                continue

            if len(cleaned) == 1 and cleaned[0].upper() == "NONE":
                value = NONE_VALUE
            else:
                value = [
                    item for item in cleaned if item.upper() != "NONE"
                ]

                if not value:
                    continue

        updates[name] = value

    return updates


# ============================================================
# Summary segments
# ============================================================

def format_value(value) -> str:

    if isinstance(value, list):
        return "، ".join(str(item) for item in value)

    return str(value)


def build_summary_segments(
    candidate: dict,
    interview_manager,
) -> list[dict]:
    """
    Assembles the full summary as a segment list: cached intro,
    then for every field the candidate actually has a value for,
    either its fully-cached "none" line or its cached label followed
    by the one genuinely dynamic piece (the value itself), then the
    cached confirm line.

    A field never mentioned at all is skipped entirely, not read
    back as empty.
    """

    segments = list(interview_manager.summary_config["intro"])

    for label in interview_manager.summary_config["labels"]:

        field_name = label["field"]

        value = candidate.get(field_name)

        if value is None:
            continue

        if isinstance(value, (list, str)) and not value:
            continue

        if value == NONE_VALUE:

            none_line = interview_manager.none_line_for_field(
                field_name
            )

            if none_line:
                segments.append(
                    {"type": "static", **none_line}
                )

            continue

        segments.append(
            {"type": "static", "id": label["id"], "text": label["text"]}
        )

        segments.append(
            {"type": "value_text", "text": format_value(value)}
        )

    segments.extend(interview_manager.summary_config["confirm"])

    return segments


# ============================================================
# Summary reply classification
# ============================================================

SUMMARY_REPLY_PROMPT = """
You are classifying an Egyptian Arabic candidate's reply to a
spoken summary of their interview answers.

Decide exactly one of two things:

1. The candidate CONFIRMED the summary is correct and wants to
   finish  -> action = "confirm"
2. The candidate wants to CORRECT one specific field
   -> action = "correct", field = <field name>
   - If they ALSO gave the new value in the same utterance,
     set new_value to that value ONLY (not the whole sentence).
   - If they only asked to change a field without saying the
     new value, leave new_value empty/null.

Always call resolve_summary_reply.

FIELDS THAT CAN BE CORRECTED:

{fields}

EXAMPLES:

"تمام و صح و شكرا"                    -> confirm
"ايوة كده تمام خلاص"                  -> confirm
"غيرلي المؤهل"                        -> correct, education_level
"لأ عايز أعدل سنين الخبرة"             -> correct, years_of_experience
"اسمي غلط"                            -> correct, candidate_name
"عدل الاسم اسمي طاهر"                 -> correct, candidate_name, new_value="طاهر"
"update the name my name is taher"    -> correct, candidate_name, new_value="taher"
"غيرلي الاسم، أنا اسمي أحمد علي"       -> correct, candidate_name, new_value="أحمد علي"
"عدل المهارات لبايثون وجافا"           -> correct, key_skills, new_value="بايثون، جافا"
"عايز أغير المجال لداتا ساينس"         -> correct, target_domain, new_value="داتا ساينس"

CANDIDATE REPLY:

{transcript}
"""


def _build_summary_reply_tool(fields: list[dict]) -> StructuredTool:

    from typing import Literal

    names = tuple(f["name"] for f in fields)

    Model = create_model(
        "SummaryReply",
        action=(
            str,
            Field(
                ...,
                description=(
                    "'confirm' to finish, or 'correct' to change "
                    "one field."
                ),
            ),
        ),
        field=(
            Literal[names] | None,
            Field(
                default=None,
                description="Required when action is 'correct'.",
            ),
        ),
        new_value=(
            str | None,
            Field(
                default=None,
                description=(
                    "The new field value if the candidate said it "
                    "in the same utterance. For list fields, join "
                    "items with an Arabic comma (،). Leave null "
                    "when they only asked to change the field."
                ),
            ),
        ),
    )

    def resolve_summary_reply(
        action: str,
        field: str | None = None,
        new_value: str | None = None,
    ):
        return {
            "action": action,
            "field": field,
            "new_value": new_value,
        }

    return StructuredTool.from_function(
        func=resolve_summary_reply,
        name="resolve_summary_reply",
        description="Record the candidate's reply to the summary.",
        args_schema=Model,
    )


def parse_inline_correction_value(
    field: dict,
    new_value: str,
) -> str | list[str] | None:
    """
    Turn the LLM's inline new_value string into the stored
    candidate shape for this field (string, list, or NONE_VALUE).
    """

    text = new_value.strip()

    if not text:
        return None

    if text.upper() == "NONE":
        return NONE_VALUE

    if field.get("multi"):

        parts = [
            part.strip()
            for part in text.replace(",", "،").split("،")
            if part.strip()
        ]

        if not parts:
            return None

        if len(parts) == 1 and parts[0].upper() == "NONE":
            return NONE_VALUE

        return [
            part for part in parts if part.upper() != "NONE"
        ] or None

    return text


async def classify_summary_reply(
    transcript: str,
    fields: list[dict],
) -> dict:
    """
    Returns:
        {"action": "confirm"}
        {"action": "correct", "field": ..., "new_value": str|None}
        {"action": "unclear"}

    When correcting, new_value is set only if the candidate gave
    the replacement in the same utterance — the caller then applies
    it without re-asking the question.
    """

    transcript = transcript.strip()

    if not transcript:
        return {"action": "unclear", "field": None, "new_value": None}

    correctable = [f for f in fields if f.get("asked_by")]

    described = "\n".join(
        f"- {f['name']} — {f['label']}" for f in correctable
    )

    tool = _build_summary_reply_tool(correctable)

    response = await llm.bind_tools([tool]).ainvoke(
        [
            SystemMessage(
                content=SUMMARY_REPLY_PROMPT.format(
                    fields=described, transcript=transcript
                )
            ),
            HumanMessage(content=transcript),
        ]
    )

    print("[SUMMARY] Tool calls:", response.tool_calls)

    if not response.tool_calls:
        return {"action": "unclear", "field": None, "new_value": None}

    args = response.tool_calls[0].get("args", {})

    action = args.get("action")
    field = args.get("field")
    new_value = args.get("new_value")

    if isinstance(new_value, str):
        new_value = new_value.strip() or None
    else:
        new_value = None

    if action == "confirm":
        return {"action": "confirm", "field": None, "new_value": None}

    if action == "correct" and field:
        return {
            "action": "correct",
            "field": field,
            "new_value": new_value,
        }

    return {"action": "unclear", "field": None, "new_value": None}