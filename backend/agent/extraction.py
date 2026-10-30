import re
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from backend import config
from backend.agent.conversation_manager import normalize_digits


# ============================================================
# LLM
# ============================================================
# temperature=0: this is extraction/classification, not writing.

llm = ChatGroq(
    api_key=config.GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0,
)


# ============================================================
# Known STT artifacts / obviously empty answers
# ============================================================

INVALID_TRANSCRIPTS = {
    "",
    "@@فراغ",
    "فراغ",
}


def is_obviously_invalid(transcript: str) -> bool:

    return transcript.strip().lower() in INVALID_TRANSCRIPTS


# ============================================================
# Slot type -> tool argument type
# ============================================================

VIA_VALUES = ("door", "open_passage", "none")


class RoomEdge(BaseModel):
    room_a: str = Field(..., description="Room id, e.g. kitchen or bedroom_1.")
    room_b: str = Field(..., description="Room id of the other room.")
    via: Literal["door", "open_passage", "none"] = Field(
        "door", description="How the two rooms connect."
    )


def _annotation(slot: dict):

    kind = slot["type"]

    if kind == "integer":
        return int | None

    if kind == "number":
        return float | None

    if kind == "boolean":
        return bool | None

    if kind == "enum":

        options = slot.get("options")

        return (Literal[tuple(options)] | None) if options else (str | None)

    if kind == "multi_enum":
        return list[Literal[tuple(slot["options"])]] | None

    if kind == "edge_list":
        return list[RoomEdge] | None

    return str | None


def _describe_slots(slots: list[dict]) -> str:

    return "\n".join(f"- {s['id']}: {s['description']}" for s in slots)


# ============================================================
# Tools
# ============================================================

def _build_update_tool(slots: list[dict]) -> StructuredTool:

    model_fields = {
        slot["id"]: (
            _annotation(slot),
            Field(default=None, description=slot["description"]),
        )
        for slot in slots
    }

    ids = tuple(s["id"] for s in slots)

    model_fields["use_default_for"] = (
        list[Literal[ids]] | None,
        Field(
            default=None,
            description=(
                "Slot ids the user said they don't care about or want "
                "you to decide (e.g. 'مش فارق', 'إنت اختار', 'you decide')."
            ),
        ),
    )

    UpdateModel = create_model("RequirementsUpdate", **model_fields)

    def update_requirements(**kwargs):
        return kwargs

    return StructuredTool.from_function(
        func=update_requirements,
        name="update_requirements",
        description=(
            "Call this when the user gave real information about the "
            "apartment they want. Fill EVERY listed field they clearly "
            "stated, including ones that answer later questions. Never "
            "call this for greetings or meta-conversation alone."
        ),
        args_schema=UpdateModel,
    )


def _build_intent_tool(intents: list[dict]) -> StructuredTool | None:

    if not intents:
        return None

    ids = tuple(i["id"] for i in intents)

    IntentModel = create_model(
        "IntentMatch",
        intent_id=(Literal[ids], Field(..., description="Which situation this is.")),
    )

    def match_intent(intent_id: str):
        return {"intent_id": intent_id}

    return StructuredTool.from_function(
        func=match_intent,
        name="match_intent",
        description=(
            "Call this when the user said something that is NOT an answer "
            "but matches one of the known situations (can't hear, asks to "
            "repeat, needs a moment, wants to stop). Never call it for a "
            "normal answer."
        ),
        args_schema=IntentModel,
    )


def _describe_intents(intents: list[dict]) -> str:

    return "\n".join(
        f"- {i['id']}: {i['description']}\n  examples: {' | '.join(i.get('examples', []))}"
        for i in intents
    )


# ============================================================
# One extraction turn
# ============================================================

TURN_PROMPT = """
You are handling one turn of a voice conversation in which an
assistant collects requirements for an apartment floor plan.
The user speaks Egyptian Arabic or English.

CURRENT QUESTION (what was just said to the user):

{question}

CURRENT QUESTION FIELDS (at least one should be filled if the user is
answering normally; empty means this is the user's first, open
description of the apartment):

{current_fields}

Decide exactly ONE of three things:

1. The user gave information about the apartment
   -> call update_requirements with EVERY field they clearly stated,
      current and later ones. Leave everything else null.
2. The user said something else that matches one of the known
   situations below -> call match_intent
3. Neither clearly applies -> call NOTHING

Never call both tools. Never guess a value that was not said.

ALL FIELDS YOU MAY FILL THIS TURN:

{fields}

KNOWN ROOMS (valid room ids for connection and room-choice fields;
empty means the room set is not final, so leave those fields null):

{rooms}

KNOWN SITUATIONS (for match_intent):

{intents}

RULES:

- Egyptian terms: "ريسبشن" = living room. "أمريكاني" or "مطبخ مفتوح" =
  open kitchen. "مطبخ مقفول" = closed kitchen. "أوضة سفرة" = dining
  room. "حمامين" = 2 bathrooms. "أوضتين" = 2 bedrooms.
- "شقة ٣ أوض" or "3 rooms" with no other qualifier means 3 bedrooms
  (the living room is counted separately).
- Numbers may be spoken as words: واحد، اتنين، تلاتة، أربعة، خمسة،
  ستة. Convert them to digits.
- Only use "use_default_for" when the user explicitly leaves the
  choice to you or says it doesn't matter.
- For extra_rooms, an explicit "no extras" / "مفيش حاجة زيادة" means an
  empty list, which is different from not mentioning extras.
- A family size does NOT tell you the number of bedrooms. Do not
  derive one from the other.
- For room connections use only the ids under KNOWN ROOMS. "open"
  or "من غير باب" = open_passage.
- The speech comes from an imperfect speech-to-text system; extract
  what is clearly there and ignore obvious noise.
- A real (even partial) answer always takes priority over an
  intent match.

USER SAID:

{transcript}
"""


async def process_turn(
    question_text: str,
    slots: list[dict],
    current_slots: list[dict],
    intents: list[dict],
    transcript: str,
    rooms_context: str = "",
    room_ids: list[str] | None = None,
) -> dict:
    """
    One LLM call, three possible outcomes:

        {"kind": "answer", "updates": {...}, "use_default_for": [...]}
        {"kind": "intent", "intent_id": "..."}
        {"kind": "none"}
    """

    update_tool = _build_update_tool(slots)
    intent_tool = _build_intent_tool(intents)

    tools = [update_tool] + ([intent_tool] if intent_tool else [])

    prompt = TURN_PROMPT.format(
        question=question_text,
        current_fields=_describe_slots(current_slots) or "(none, open description)",
        fields=_describe_slots(slots),
        rooms=rooms_context or "(not final yet)",
        intents=_describe_intents(intents) or "(none)",
        transcript=transcript,
    )

    response = await llm.bind_tools(tools).ainvoke(
        [SystemMessage(content=prompt), HumanMessage(content=transcript)]
    )

    print("[EXTRACTION] Tool calls:", response.tool_calls)

    for tool_call in response.tool_calls or []:

        args = tool_call.get("args", {})

        if tool_call["name"] == "match_intent" and args.get("intent_id"):
            return {"kind": "intent", "intent_id": args["intent_id"]}

        if tool_call["name"] == "update_requirements":

            updates = _clean_updates(args, slots, set(room_ids or []))

            allowed = {s["id"] for s in slots}
            defaults = [d for d in (args.get("use_default_for") or []) if d in allowed]

            if updates or defaults:
                return {"kind": "answer", "updates": updates, "use_default_for": defaults}

    return {"kind": "none"}


# ============================================================
# Cleaning: never trust the model's output blindly
# ============================================================

def _clean_updates(args: dict, slots: list[dict], room_ids: set[str]) -> dict:

    by_id = {s["id"]: s for s in slots}
    updates = {}

    for sid, value in args.items():

        slot = by_id.get(sid)

        if slot is None or value is None:
            continue

        cleaned = _clean_value(slot, value, room_ids)

        if cleaned is not None:
            updates[sid] = cleaned

    return updates


def _clean_value(slot: dict, value, room_ids: set[str]):

    kind = slot["type"]

    if kind in ("integer", "number"):

        try:
            number = float(normalize_digits(str(value)))
        except ValueError:
            return None

        if "min" in slot and number < slot["min"]:
            return None

        if "max" in slot and number > slot["max"]:
            return None

        return int(round(number)) if kind == "integer" else number

    if kind == "boolean":
        return value if isinstance(value, bool) else None

    if kind == "enum":

        text = str(value).strip()
        options = slot.get("options")

        if options:
            return text if text in options else None

        return text if text in room_ids else None

    if kind == "multi_enum":

        if not isinstance(value, list):
            return None

        items = [
            str(i).strip()
            for i in value
            if str(i).strip() and str(i).strip().upper() != "NONE"
        ]

        valid = [i for i in dict.fromkeys(items) if i in slot["options"]]

        # [] is a real answer ("no extras"); a non-empty list with
        # nothing valid in it is just noise.
        return valid if valid or not items else None

    if kind == "edge_list":

        if not isinstance(value, list):
            return None

        edges = []

        for item in value:

            if hasattr(item, "model_dump"):
                item = item.model_dump()

            if not isinstance(item, dict):
                continue

            a, b = item.get("room_a"), item.get("room_b")
            via = item.get("via", "door")

            if a in room_ids and b in room_ids and a != b and via in VIA_VALUES:
                edges.append({"from": a, "to": b, "via": via})

        return edges or None

    return None


# ============================================================
# Reply to a yes/no style question (summary, soft warnings)
# ============================================================

CONFIRM_PROMPT = """
You are classifying a user's reply to something an assistant just said
while collecting requirements for an apartment floor plan. The user
speaks Egyptian Arabic or English.

WHAT THE ASSISTANT JUST SAID:

{context}

Decide exactly one of two things:

1. The user ACCEPTS it as it is / says it is fine and wants to move on
   -> action = "confirm"
2. The user wants to CHANGE one specific field
   -> action = "correct", field = <field name>
   - If they ALSO gave the new value in the same utterance, set
     new_value to that value ONLY (not the whole sentence).
   - If they only asked to change it, leave new_value null.

Always call resolve_reply.

FIELDS THAT CAN BE CORRECTED:

{fields}

EXAMPLES:

"تمام كده صح"                         -> confirm
"أيوه كده كويس كمل"                   -> confirm
"غير عدد الحمامات"                    -> correct, num_bathrooms
"خليهم ٣ أوض نوم"                     -> correct, num_bedrooms, new_value="3"
"المطبخ يبقى مفتوح"                   -> correct, kitchen_type, new_value="open"
"عايز أغير الوصلات"                   -> correct, room_connections
"make it two bathrooms"               -> correct, num_bathrooms, new_value="2"

USER REPLY:

{transcript}
"""


def _build_reply_tool(fields: list[dict]) -> StructuredTool:

    names = tuple(f["name"] for f in fields)

    Model = create_model(
        "ConfirmReply",
        action=(
            Literal["confirm", "correct"],
            Field(..., description="'confirm' to accept, 'correct' to change one field."),
        ),
        field=(
            Literal[names] | None,
            Field(default=None, description="Required when action is 'correct'."),
        ),
        new_value=(
            str | None,
            Field(
                default=None,
                description=(
                    "New value if said in the same utterance. For list "
                    "fields join items with an Arabic comma. Null when "
                    "they only asked to change the field."
                ),
            ),
        ),
    )

    def resolve_reply(action: str, field: str | None = None, new_value: str | None = None):
        return {"action": action, "field": field, "new_value": new_value}

    return StructuredTool.from_function(
        func=resolve_reply,
        name="resolve_reply",
        description="Record the user's reply.",
        args_schema=Model,
    )


async def classify_confirmation_reply(
    transcript: str,
    context_text: str,
    correctable: list[dict],
) -> dict:
    """
    Returns:
        {"action": "confirm"}
        {"action": "correct", "field": ..., "new_value": str | None}
        {"action": "unclear"}
    """

    transcript = transcript.strip()

    if not transcript:
        return {"action": "unclear", "field": None, "new_value": None}

    described = "\n".join(f"- {f['name']} ({f['label']})" for f in correctable)

    response = await llm.bind_tools([_build_reply_tool(correctable)]).ainvoke(
        [
            SystemMessage(
                content=CONFIRM_PROMPT.format(
                    context=context_text, fields=described, transcript=transcript
                )
            ),
            HumanMessage(content=transcript),
        ]
    )

    print("[CONFIRM] Tool calls:", response.tool_calls)

    if not response.tool_calls:
        return {"action": "unclear", "field": None, "new_value": None}

    args = response.tool_calls[0].get("args", {})

    action, field = args.get("action"), args.get("field")
    new_value = args.get("new_value")
    new_value = new_value.strip() if isinstance(new_value, str) and new_value.strip() else None

    if action == "confirm":
        return {"action": "confirm", "field": None, "new_value": None}

    if action == "correct" and field:
        return {"action": "correct", "field": field, "new_value": new_value}

    return {"action": "unclear", "field": None, "new_value": None}


# ============================================================
# Inline correction: "make it 3 bedrooms" -> 3
# ============================================================

YES_WORDS = {"yes", "true", "أيوه", "ايوه", "أيوة", "ايوة", "اه", "آه", "نعم", "أيوا"}
NO_WORDS = {"no", "false", "لا", "لأ", "لاء"}


def parse_inline_correction_value(manager, slot: dict, text: str, room_ids: list[str]):
    """
    Turn the LLM's new_value string into the stored value for this
    slot, or None if it can't be parsed (the caller then simply asks
    the slot's own question again).
    """

    text = normalize_digits(text.strip())

    if not text:
        return None

    kind = slot["type"]

    if kind in ("integer", "number"):

        match = re.search(r"\d+(?:\.\d+)?", text)

        if not match:
            return None

        number = float(match.group())

        if "min" in slot and number < slot["min"]:
            return None

        if "max" in slot and number > slot["max"]:
            return None

        return int(number) if kind == "integer" else number

    if kind == "boolean":

        lowered = text.lower()

        if lowered in YES_WORDS:
            return True

        if lowered in NO_WORDS:
            return False

        return None

    if kind == "enum":

        if slot.get("options"):
            return manager.match_option(slot, text)

        return text if text in room_ids else None

    if kind == "multi_enum":

        parts = [p.strip() for p in text.replace(",", "،").split("،") if p.strip()]
        matched = [manager.match_option(slot, p) for p in parts]

        if parts and all(matched):
            return list(dict.fromkeys(matched))

        return None

    return None