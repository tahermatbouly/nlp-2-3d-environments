import hashlib
import json
from pathlib import Path


# Stored when the candidate explicitly says they don't have
# something ("مفيش مهارات"). Distinct from None/missing: the
# question WAS answered, so we must never keep re-asking it.
NONE_VALUE = "__none__"


def value_cache_id(text: str) -> str:
    """
    Cache key for a dynamic "value" segment (a name, a number, a
    skill list). These aren't in questions.json — they're whatever
    the candidate said — but the same values recur across many
    calls (common names, "سنتين", "بكالوريوس حاسبات"), so caching by
    a hash of the exact text still turns most of them into cache
    hits after the first few calls, not just the first.
    """

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    return f"value__{digest}"


class InterviewManager:
    """
    Loads the segment-based sequential interview config.

    Every spoken line is a list of segments:
        {"type": "static", "id": ..., "text": ...}  - stable id,
            pre-generated once, ever, regardless of candidate.
        {"type": "value", "field": ...}              - resolved from
            state["candidate"][field] at speak time; short, cached
            by its own text hash rather than a fixed id.
        {"type": "live", "template": "...{field}..."} - whole phrase
            synthesized fresh every time (no disk cache); used for
            natural personalized greetings.

    This is what lets the end-of-interview summary be almost
    entirely cached even though its content depends on the
    candidate: only the values are ever synthesized fresh.
    """

    def __init__(self, config_path: str):

        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        settings = config.get("settings", {})

        self.silence_step_seconds = settings.get(
            "silence_step_seconds", 5
        )
        self.max_retries_per_question = settings.get(
            "max_retries_per_question", 3
        )
        self.skip_optional_after_max_retries = settings.get(
            "skip_optional_after_max_retries", True
        )

        self.fields = config["fields"]
        self._fields_by_name = {f["name"]: f for f in self.fields}

        self.questions = config["questions"]
        self._questions_by_id = {q["id"]: q for q in self.questions}

        self.summary_config = config["summary"]
        self.completion_segments = config["completion"]

        self.silence_prompts = config.get("silence_prompts", [])

        self.system_messages = {
            item["id"]: item
            for item in config.get("system_messages", [])
        }

        self.intents = config.get("intents", [])
        self._intents_by_id = {i["id"]: i for i in self.intents}

        # Precomputed static "none" summary line per field — e.g.
        # "المهارات: حضرتك قلت إن مفيش مهارات" — combining the
        # field's label, the shared none_prefix, and that field's
        # own none_text into ONE fully static, fully cacheable clip.
        # This means an explicit "I don't have any" in the summary
        # never needs live synthesis either, same as everything
        # else.
        none_prefix_text = self.summary_config["none_prefix"]["text"]

        self._none_line_by_field = {}

        for field in self.fields:

            label = self._label_for_field(field["name"])

            if label is None:
                continue

            text = f"{label} {none_prefix_text} {field['none_text']}"

            self._none_line_by_field[field["name"]] = {
                "id": f"summary__none_line__{field['name']}",
                "text": text,
            }

    # ========================================================
    # QUESTIONS
    # ========================================================

    def get_question(self, index: int) -> dict | None:

        if index < 0 or index >= len(self.questions):
            return None

        return self.questions[index]

    def get_question_by_id(self, question_id: str) -> dict | None:

        return self._questions_by_id.get(question_id)

    def target_fields(self, question: dict) -> list[str]:

        if "target_field" in question:
            return [question["target_field"]]

        return question.get("target_fields", [])

    def is_finished(self, index: int) -> bool:

        return index >= len(self.questions)

    # ========================================================
    # FIELDS
    # ========================================================

    def get_field(self, name: str) -> dict | None:

        return self._fields_by_name.get(name)

    def question_for_field(self, field_name: str) -> dict | None:
        """
        Which question asks for this field — used to jump straight
        there when the candidate corrects one field after the
        summary. Looks at "asked_by" first (explicit), falling back
        to a scan of every question's target fields.
        """

        field = self._fields_by_name.get(field_name)

        if field and field.get("asked_by"):
            return self._questions_by_id.get(field["asked_by"])

        for question in self.questions:

            if field_name in self.target_fields(question):
                return question

        return None

    def _label_for_field(self, field_name: str) -> str | None:

        for label in self.summary_config["labels"]:

            if label["field"] == field_name:
                return label["text"]

        return None

    def none_line_for_field(self, field_name: str) -> dict | None:

        return self._none_line_by_field.get(field_name)

    # ========================================================
    # SYSTEM MESSAGES / INTENTS / SILENCE
    # ========================================================

    def get_system_message(self, message_id: str) -> dict | None:

        return self.system_messages.get(message_id)

    def get_intent(self, intent_id: str) -> dict | None:

        return self._intents_by_id.get(intent_id)

    def get_silence_prompt(self, level: int) -> dict | None:
        """
        level is 1-based (first nudge = 1). Returns None once past
        the configured escalation steps.
        """

        index = level - 1

        if index < 0 or index >= len(self.silence_prompts):
            return None

        return self.silence_prompts[index]

    # ========================================================
    # TTS PRE-GENERATION
    # ========================================================

    def static_tts_items(self) -> list[tuple[str, str]]:
        """
        Every (id, text) pair that can be pre-generated once, ever
        — everything except the candidate's own extracted values.
        """

        items: list[tuple[str, str]] = []

        def collect(segments):
            for seg in segments:
                if seg["type"] == "static":
                    items.append((seg["id"], seg["text"]))

        for question in self.questions:

            collect(question["segments"])
            collect(question.get("repeat_segments", []))

        collect(self.summary_config.get("processing", []))
        collect(self.summary_config["intro"])
        collect(self.summary_config["confirm"])

        for label in self.summary_config["labels"]:
            items.append((label["id"], label["text"]))

        items.append(
            (
                self.summary_config["none_prefix"]["id"],
                self.summary_config["none_prefix"]["text"],
            )
        )

        for none_line in self._none_line_by_field.values():
            items.append((none_line["id"], none_line["text"]))

        collect(self.completion_segments)

        for prompt in self.silence_prompts:
            items.append((prompt["id"], prompt["text"]))

        for message in self.system_messages.values():
            items.append((message["id"], message["text"]))

        for intent in self.intents:
            items.append((intent["id"], intent["text"]))

        return items