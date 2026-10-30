import json
import operator
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_LANGUAGE = "ar_eg"

FRONT_DOOR = "front_door"
FRONT_DOOR_NAME = {"en": "front door", "ar_eg": "باب الشقة"}

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

LIST_SEPARATOR = {"en": ", ", "ar_eg": "، "}

# Slot groups whose answers decide WHICH rooms exist. Connection
# questions wait until all of these are settled (rooms_known).
ROOM_GROUPS = {"rooms", "kitchen_living", "extras"}

# Threshold behind the "area_per_bedroom" validation rule.
MIN_AREA_PER_ROOM_M2 = 14

_COMPARE = {
    "gte": operator.ge,
    "gt": operator.gt,
    "lte": operator.le,
    "lt": operator.lt,
}


def normalize_digits(text: str) -> str:
    """Arabic-Indic digits to ASCII digits ("٣" -> "3")."""

    return text.translate(ARABIC_DIGITS)


class ConversationManager:
    """
    Loads requirements_questions.json and answers every question the
    graph has about it:

      - which slot(s) to ask next (missing, applicable, in order)
      - what the rooms are, and how they connect
      - which validation rule is failing
      - what to say (as segments handler.py can play)

    The LLM is never involved here. Everything is deterministic, so
    the question flow can be tested without any model.

    Segments use the same two kinds as before:
        {"kind": "static", "id": ..., "text": ...}  cacheable TTS clip
        {"kind": "live", "text": ...}               synthesized fresh
    """

    def __init__(self, config_path: str, default_language: str = DEFAULT_LANGUAGE):

        with open(Path(config_path), "r", encoding="utf-8") as f:
            config = json.load(f)

        self.default_language = default_language
        self.languages = config.get("languages", ["en", "ar_eg"])

        self.policy = config["policy"]
        self.messages = config.get("messages", {})
        self.room_types = config["room_types"]
        self.connection_types = config["connection_types"]
        self.default_rules = config.get("default_connection_rules", [])
        self.validation_rules = config.get("validation_rules", [])

        self.intents = config.get("intents", [])
        self._intents_by_id = {i["id"]: i for i in self.intents}

        self.max_questions_per_turn = self.policy.get("max_questions_per_turn", 2)
        self.max_reprompts = self.policy.get("max_reprompts_per_slot", 2)
        self._optional_to_ask = set(self.policy.get("optional_slots_to_ask", []))

        self.slots = self._flatten_slots(config["slots"])
        self._slots_by_id = {s["id"]: s for s in self.slots}

    # ========================================================
    # LOADING
    # ========================================================

    def _flatten_slots(self, raw_slots: list[dict]) -> list[dict]:
        """
        Followups of a slot (bedroom_access, bathroom_access) become
        ordinary slots that only apply in manual connection mode,
        after room_connections has been answered. Slots are then
        ordered by policy.ask_order (stable inside a group).
        """

        flat: list[dict] = []

        for slot in raw_slots:

            flat.append(dict(slot))

            for followup in slot.get("followups", []):

                sub = dict(followup)
                sub.update(
                    {
                        "group": slot["group"],
                        "required": False,
                        "type": "enum",
                        "default": None,
                        "followup_of": slot["id"],
                        "ask_when": [
                            {"slot": slot["id"], "op": "resolved"},
                            {"slot": "connections_mode", "op": "eq", "value": "manual"},
                            *followup.get("ask_when", []),
                        ],
                    }
                )

                self._optional_to_ask.add(sub["id"])
                flat.append(sub)

        order = {g: i for i, g in enumerate(self.policy.get("ask_order", []))}
        flat.sort(key=lambda s: order.get(s["group"], len(order)))

        for slot in flat:
            slot["description"] = self._describe(slot)

        return flat

    def _describe(self, slot: dict) -> str:
        """English description handed to the extraction LLM."""

        label = self.text(self.messages.get("summary_labels", {}).get(slot["id"]), "en")
        parts = [f"{label or slot['id']}. Type: {slot['type']}."]

        if slot.get("options"):
            parts.append("Allowed values: " + ", ".join(slot["options"]) + ".")

        if slot.get("options_from") == "rooms_list":
            parts.append("Must be one of the room ids listed under KNOWN ROOMS.")

        if "min" in slot or "max" in slot:
            parts.append(f"Range: {slot.get('min', '-')} to {slot.get('max', '-')}.")

        if slot["type"] == "edge_list":
            parts.append(
                "List of room pairs (room ids) and how they connect: "
                "door, open_passage, or none."
            )

        hints = slot.get("extraction_hints")

        if hints:
            keywords = hints.get("en", []) + hints.get("ar_eg", [])
            parts.append("Keywords: " + ", ".join(keywords) + ".")

        return " ".join(parts)

    # ========================================================
    # TEXT HELPERS
    # ========================================================

    def text(self, obj: Any, lang: str | None = None) -> str:

        if obj is None:
            return ""

        if isinstance(obj, str):
            return obj

        lang = lang or self.default_language

        return obj.get(lang) or obj.get(self.default_language) or obj.get("en") or ""

    def get_slot(self, slot_id: str) -> dict | None:

        return self._slots_by_id.get(slot_id)

    def slot_label(self, slot_id: str, lang: str) -> str:

        return self.text(self.messages.get("summary_labels", {}).get(slot_id), lang) or slot_id

    def correctable_slots(self) -> list[dict]:
        """Slots the user can ask to change after the summary."""

        labels = self.messages.get("summary_labels", {})

        return [
            {"name": s["id"], "label": self.text(labels[s["id"]], "en")}
            for s in self.slots
            if s["id"] in labels and s.get("question")
        ]

    def option_name(self, option: str, lang: str) -> str:

        names = self.messages.get("option_names", {})

        if option in names:
            return self.text(names[option], lang)

        if option in self.room_types:
            return self.text(self.room_types[option]["name"], lang)

        return option

    def match_option(self, slot: dict, text: str) -> str | None:
        """Map free text ("أمريكاني", "open") to one of slot['options']."""

        needle = text.strip().lower()

        if not needle:
            return None

        options = slot.get("options", [])

        def names(option: str) -> list[str]:
            return [option.lower()] + [
                self.option_name(option, lang).lower() for lang in self.languages
            ]

        for option in options:
            if needle in names(option):
                return option

        for option in options:
            if any(n and (needle in n or n in needle) for n in names(option)):
                return option

        return None

    # ========================================================
    # CONDITIONS / WHICH SLOT NEXT
    # ========================================================

    def _is_askable(self, slot: dict) -> bool:
        """Required slots and the flagged optional ones are asked."""

        return bool(slot.get("required")) or slot["id"] in self._optional_to_ask

    def conditions_met(self, slot: dict, req: dict, resolved: list[str]) -> bool:

        for cond in slot.get("ask_when", []):

            sid, op = cond["slot"], cond["op"]
            target = cond.get("value")

            value = (
                self.rooms_known(req, resolved)
                if sid == "rooms_known"
                else req.get(sid)
            )

            if op == "resolved":
                ok = sid in resolved
            elif op == "missing":
                ok = value is None
            elif op == "eq":
                ok = value == target
            elif op == "ne":
                ok = value != target
            elif op == "contains":
                ok = isinstance(value, list) and target in value
            elif op in _COMPARE:
                ok = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and _COMPARE[op](value, target)
                )
            else:
                ok = False

            if not ok:
                return False

        return True

    def rooms_known(self, req: dict, resolved: list[str]) -> bool:
        """True once every slot that decides the room set is settled."""

        for slot in self.slots:

            if slot["group"] not in ROOM_GROUPS or not self._is_askable(slot):
                continue

            if slot["id"] in resolved:
                continue

            if self.conditions_met(slot, req, resolved):
                return False

        return True

    def next_slots(self, req: dict, resolved: list[str]) -> list[str]:
        """Up to max_questions_per_turn slot ids still to ask."""

        pending = []

        for slot in self.slots:

            if slot["id"] in resolved or not self._is_askable(slot):
                continue

            if not self.conditions_met(slot, req, resolved):
                continue

            pending.append(slot["id"])

            if len(pending) >= self.max_questions_per_turn:
                break

        return pending

    def extractable_slots(
        self, req: dict, resolved: list[str], current_ids: list[str]
    ) -> list[dict]:
        """Everything the LLM may fill this turn: the slots being asked
        plus any still-unresolved slot the user volunteers early."""

        return [
            s
            for s in self.slots
            if s["id"] in current_ids or s["id"] not in resolved
        ]

    def fill_defaults(self, req: dict, resolved: list[str], assumed: list[str]) -> None:
        """Settle applicable slots we never ask about, using defaults."""

        for slot in self.slots:

            sid = slot["id"]

            if sid in resolved or self._is_askable(slot):
                continue

            if not self.conditions_met(slot, req, resolved):
                continue

            default = slot.get("default")

            if default is None:
                continue

            if isinstance(default, str) and default.startswith("apply_"):
                default = []

            req[sid] = deepcopy(default)
            resolved.append(sid)
            assumed.append(sid)

    def settle_with_default(
        self, slot_id: str, req: dict, resolved: list[str], assumed: list[str]
    ) -> bool:
        """Use a slot's default as its value. False if it has none."""

        default = self._slots_by_id[slot_id].get("default")

        if default is None:
            return False

        if isinstance(default, str) and default.startswith("apply_"):
            default = []

        req[slot_id] = deepcopy(default)

        if slot_id not in resolved:
            resolved.append(slot_id)

        if slot_id not in assumed:
            assumed.append(slot_id)

        return True

    # ========================================================
    # ROOMS
    # ========================================================

    def rooms(self, req: dict) -> list[dict]:
        """The rooms implied by the answers so far."""

        has_living = req.get("has_living_room")

        if has_living is None:
            has_living = req.get("property_type") != "studio"

        extras = req.get("extra_rooms") or []
        found: list[tuple[str, str]] = []

        if has_living:
            found.append(("living", "living"))

        found.append(("kitchen", "kitchen"))

        for i in range(1, (req.get("num_bedrooms") or 0) + 1):
            found.append((f"bedroom_{i}", "bedroom"))

        for i in range(1, (req.get("num_bathrooms") or 0) + 1):
            found.append((f"bathroom_{i}", "bathroom"))

        if "balcony" in extras:
            for i in range(1, (req.get("num_balconies") or 1) + 1):
                found.append((f"balcony_{i}", "balcony"))

        for room_type in ("dining", "storage", "laundry"):
            if room_type in extras:
                found.append((room_type, room_type))

        if req.get("bedroom_access") == "hallway":
            found.append(("hallway", "hallway"))

        counts = Counter(t for _, t in found)

        return [
            {
                "id": rid,
                "type": rtype,
                "count": counts[rtype],
                "index": int(rid.rsplit("_", 1)[1]) if rid[-1].isdigit() else 1,
            }
            for rid, rtype in found
        ]

    def room_label(self, req: dict, room_id: str, lang: str) -> str:

        if room_id == FRONT_DOOR:
            return self.text(FRONT_DOOR_NAME, lang)

        for room in self.rooms(req):

            if room["id"] == room_id:

                name = self.text(self.room_types[room["type"]]["name"], lang)

                return f"{name} {room['index']}" if room["count"] > 1 else name

        return room_id

    def room_ids(self, req: dict, resolved: list[str]) -> list[str]:
        """Valid room ids for connection answers. Empty until the room
        set is final, so the LLM can't connect rooms that don't exist."""

        if not self.rooms_known(req, resolved):
            return []

        return [r["id"] for r in self.rooms(req)]

    def rooms_list_text(self, req: dict, lang: str) -> str:

        return LIST_SEPARATOR[lang if lang in LIST_SEPARATOR else "en"].join(
            self.room_label(req, r["id"], lang) for r in self.rooms(req)
        )

    def rooms_context(self, req: dict, resolved: list[str], lang: str) -> str:
        """KNOWN ROOMS block for the extraction prompt."""

        ids = self.room_ids(req, resolved)

        if not ids:
            return ""

        return "\n".join(
            f"- {rid} ({self.room_label(req, rid, lang)})" for rid in ids
        )

    def entrance_room(self, req: dict) -> str:

        ids = {r["id"] for r in self.rooms(req)}
        entrance = req.get("entrance_room")

        if entrance in ids:
            return entrance

        return "living" if "living" in ids else self.hub(req)

    def hub(self, req: dict) -> str:
        """The room everything else opens onto by default."""

        ids = [r["id"] for r in self.rooms(req)]

        if "living" in ids:
            return "living"

        bedrooms = [i for i in ids if i.startswith("bedroom_")]

        return bedrooms[0] if bedrooms else "kitchen"

    def reconcile(self, req: dict, resolved: list[str], assumed: list[str]) -> None:
        """
        After any change, drop answers that point at rooms that no
        longer exist (e.g. user went from 3 bedrooms to 2), and
        un-settle those slots so they are asked again.
        """

        ids = {r["id"] for r in self.rooms(req)}

        def unresolve(sid: str) -> None:

            req.pop(sid, None)

            if sid in resolved:
                resolved.remove(sid)

            if sid in assumed:
                assumed.remove(sid)

        edges = req.get("room_connections")

        if isinstance(edges, list):

            kept = [e for e in edges if e["from"] in ids and e["to"] in ids]

            if len(kept) != len(edges):
                req["room_connections"] = kept

        entrance = req.get("entrance_room")

        if entrance is not None and entrance not in ids:
            unresolve("entrance_room")

        if req.get("bedroom_access") is not None and (
            (req["bedroom_access"] == "living" and "living" not in ids)
            or (req.get("num_bedrooms") or 0) < 2
        ):
            unresolve("bedroom_access")

        access = req.get("bathroom_access")

        if access is not None and access not in ids:
            unresolve("bathroom_access")

    # ========================================================
    # CONNECTION GRAPH
    # ========================================================

    def build_edges(self, req: dict) -> list[dict]:
        """
        Final list of connections.

          auto   -> default_connection_rules, then the user's own
                    edges on top (they win)
          manual -> the user's edges, then the front door, then the
                    bedroom_access / bathroom_access answers fill in
                    rooms the user left unconnected
        """

        rooms = self.rooms(req)
        ids = [r["id"] for r in rooms]
        known = set(ids)

        has_living = "living" in known
        bedrooms = [i for i in ids if i.startswith("bedroom_")]
        bathrooms = [i for i in ids if i.startswith("bathroom_")]
        balconies = [i for i in ids if i.startswith("balcony_")]

        hub = self.hub(req)
        bed_hub = "hallway" if "hallway" in known else hub
        entrance = self.entrance_room(req)

        ensuite = (
            "bathroom_1"
            if req.get("master_has_ensuite") and bathrooms and bedrooms
            else None
        )

        edges: dict[frozenset, dict] = {}

        def add(a: str | None, b: str | None, via: str = "door") -> None:

            if not a or not b or a == b:
                return

            for node in (a, b):
                if node != FRONT_DOOR and node not in known:
                    return

            key = frozenset((a, b))

            if via == "none":
                edges.pop(key, None)
            else:
                edges[key] = {"from": a, "to": b, "via": via}

        def degree(node: str) -> int:
            return sum(1 for key in edges if node in key)

        if "hallway" in known and has_living:
            add("hallway", "living", "open_passage")

        mode = req.get("connections_mode") or "auto"

        if mode == "auto":

            for rule in self.default_rules:

                frm, to = rule["from"], rule["to"]
                via = rule.get("via", "door")

                if frm == FRONT_DOOR:
                    add(FRONT_DOOR, entrance, via)

                elif frm == "living" and to == "kitchen":

                    kitchen_via = rule.get("via_by_kitchen_type", {}).get(
                        req.get("kitchen_type") or "closed", "door"
                    )

                    add("living" if has_living else hub, "kitchen", kitchen_via)

                elif frm == "living":
                    add("living", to, via)

                elif frm == "bedroom":
                    for room in bedrooms:
                        add(room, bed_hub, via)

                elif frm == "bathroom":
                    for room in bathrooms:
                        if room != ensuite:
                            add(room, bed_hub, via)

                elif frm == "master_bedroom":
                    add("bedroom_1", ensuite, via)

                elif frm == "balcony":

                    for n, room in enumerate(balconies):

                        if n == 0:
                            target = "living" if has_living else (bedrooms[0] if bedrooms else hub)
                        else:
                            target = bedrooms[n - 1] if n - 1 < len(bedrooms) else hub

                        add(room, target, via)

        for edge in req.get("room_connections") or []:
            add(edge["from"], edge["to"], edge.get("via", "door"))

        if mode == "manual":

            add(FRONT_DOOR, entrance)

            if ensuite and degree(ensuite) == 0:
                add("bedroom_1", ensuite)

            for room in bedrooms:
                if degree(room) == 0:
                    add(room, bed_hub)

            bath_target = req.get("bathroom_access") or bed_hub

            for room in bathrooms:
                if degree(room) == 0:
                    add(room, bath_target)

        return list(edges.values())

    def auto_connect(self, req: dict, room_id: str) -> None:
        """Last resort when the user can't say where a room opens."""

        known = [r["id"] for r in self.rooms(req)]

        known_set = set(known)
        target = "hallway" if "hallway" in known_set else self.hub(req)

        if target == room_id:
            target = next((r for r in known if r != room_id), None)

        if target:
            req.setdefault("room_connections", []).append(
                {"from": room_id, "to": target, "via": "door"}
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self, req: dict, acknowledged: list[str]) -> dict | None:
        """
        First failing rule from validation_rules, or None.

        kind:
          "reask"   - re-ask a slot (value is out of range)
          "warning" - plausible but suspicious, user confirms or edits
          "connect" - a room is isolated/unreachable, ask where it opens
        """

        for rule in self.validation_rules:

            rule_id = rule["id"]

            if rule_id in acknowledged:
                continue

            check = getattr(self, f"_check_{rule_id}", None)

            if check is None:
                continue

            issue = check(req)

            if issue:
                issue["rule"] = rule_id
                return issue

        return None

    def _check_need_a_bathroom(self, req: dict) -> dict | None:

        if "num_bathrooms" in req and (req["num_bathrooms"] or 0) < 1:
            return {"kind": "reask", "slot": "num_bathrooms"}

        return None

    def _check_bathrooms_not_exceed_rooms(self, req: dict) -> dict | None:

        if (req.get("num_bathrooms") or 0) > (req.get("num_bedrooms") or 0) + 2:
            return {"kind": "warning", "room": None}

        return None

    def _check_area_per_bedroom(self, req: dict) -> dict | None:

        area = req.get("total_area_m2")

        if area and area / ((req.get("num_bedrooms") or 0) + 2) < MIN_AREA_PER_ROOM_M2:
            return {"kind": "warning", "room": None}

        return None

    def _adjacency(self, req: dict) -> tuple[dict, list[str]]:

        adjacency: dict[str, set] = defaultdict(set)

        for edge in self.build_edges(req):
            adjacency[edge["from"]].add(edge["to"])
            adjacency[edge["to"]].add(edge["from"])

        return adjacency, [r["id"] for r in self.rooms(req)]

    def _check_no_isolated_room(self, req: dict) -> dict | None:

        adjacency, ids = self._adjacency(req)

        for room in ids:
            if not adjacency[room]:
                return {"kind": "connect", "room": room}

        return None

    def _check_all_rooms_reachable(self, req: dict) -> dict | None:

        adjacency, ids = self._adjacency(req)

        seen = {FRONT_DOOR}
        queue = deque([FRONT_DOOR])

        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        for room in ids:
            if room not in seen:
                return {"kind": "connect", "room": room}

        return None

    def repair_slot(self) -> dict:
        """Synthetic slot used to read 'which room should it open to'."""

        return {
            "id": "repair_target",
            "type": "enum",
            "options_from": "rooms_list",
            "description": (
                "The room the user says the isolated room opens to. "
                "Must be one of the room ids under KNOWN ROOMS."
            ),
        }

    # ========================================================
    # SEGMENTS (what gets spoken)
    # ========================================================

    def _static(self, seg_id: str, text: str) -> dict:
        return {"kind": "static", "id": seg_id, "text": text}

    def message_segments(self, key: str, lang: str) -> list[dict]:

        return [self._static(f"msg__{key}__{lang}", self.text(self.messages[key], lang))]

    def question_segments(
        self, slot_ids: list[str], req: dict, lang: str, repeat: bool = False
    ) -> list[dict]:
        """One question per slot, joined by a short 'Also,' clip."""

        segments: list[dict] = []

        for position, sid in enumerate(slot_ids):

            slot = self._slots_by_id[sid]

            use_reprompt = repeat and slot.get("reprompt")
            text = self.text(slot["reprompt"] if use_reprompt else slot["question"], lang)
            prefix = "rq" if use_reprompt else "q"

            if position > 0:
                segments.extend(self.message_segments("joiner", lang))

            if "{rooms_list}" in text:
                segments.append(
                    {
                        "kind": "live",
                        "text": text.replace("{rooms_list}", self.rooms_list_text(req, lang)),
                    }
                )
            else:
                segments.append(self._static(f"{prefix}__{sid}__{lang}", text))

        return segments

    def validation_segments(self, issue: dict, req: dict, lang: str) -> list[dict]:

        rule = next(r for r in self.validation_rules if r["id"] == issue["rule"])
        text = self.text(rule["message"], lang)

        if issue["kind"] == "connect":

            label = self.room_label(req, issue["room"], lang)

            return [{"kind": "live", "text": text.replace("{room}", label)}]

        return [self._static(f"val__{rule['id']}__{lang}", text)]

    def intent_segments(self, intent: dict, lang: str) -> list[dict]:

        return [self._static(f"intent__{intent['id']}__{lang}", self.text(intent["text"], lang))]

    def get_intent(self, intent_id: str) -> dict | None:

        return self._intents_by_id.get(intent_id)

    def confirmation_segments(self, summary: str, lang: str) -> list[dict]:

        template = self.text(self.policy["confirmation_question"], lang)

        return [{"kind": "live", "text": template.replace("{summary}", summary)}]

    # ========================================================
    # SUMMARY
    # ========================================================

    def format_value(self, slot: dict, value: Any, req: dict, lang: str) -> str:

        kind = slot["type"]

        if kind == "boolean":
            return self.text(self.messages["yes" if value else "no"], lang)

        if kind in ("integer", "number"):

            text = f"{value:g}" if isinstance(value, float) else str(value)
            unit = slot.get("unit")

            if unit:
                text += " " + self.text(self.messages.get("units", {}).get(unit), lang)

            return text

        if kind == "enum":

            if slot.get("options_from") == "rooms_list":
                return self.room_label(req, value, lang)

            return self.option_name(value, lang)

        if kind == "multi_enum":

            if not value:
                return self.text(self.messages["none"], lang)

            return LIST_SEPARATOR[lang if lang in LIST_SEPARATOR else "en"].join(
                self.option_name(v, lang) for v in value
            )

        if kind == "edge_list":

            parts = []

            for edge in value:
                via = self.text(self.connection_types.get(edge.get("via", "door")), lang)
                parts.append(
                    f"{self.room_label(req, edge['from'], lang)} - "
                    f"{self.room_label(req, edge['to'], lang)} ({via})"
                )

            return LIST_SEPARATOR[lang if lang in LIST_SEPARATOR else "en"].join(parts)

        return str(value)

    def summary_text(self, req: dict, assumed: list[str], lang: str) -> str:

        lines = []
        tag = self.text(self.messages.get("assumed_tag"), lang)

        for slot in self.slots:

            sid = slot["id"]
            value = req.get(sid)

            if value is None or sid not in self.messages.get("summary_labels", {}):
                continue

            if slot["type"] == "edge_list" and not value:
                continue

            line = f"{self.slot_label(sid, lang)}: {self.format_value(slot, value, req, lang)}"

            if sid in assumed and tag:
                line += f" {tag}"

            lines.append(line)

        return ". ".join(lines)

    # ========================================================
    # FINAL OUTPUT (input of Phase 2)
    # ========================================================

    def build_room_graph(self, req: dict, assumed: list[str]) -> dict:

        return {
            "schema_version": "1.0",
            "property_type": req.get("property_type", "apartment"),
            "total_area_m2": req.get("total_area_m2"),
            "family_size": req.get("family_size"),
            "budget_egp": req.get("budget_egp"),
            "entrance_room": self.entrance_room(req),
            "rooms": [
                {
                    "id": r["id"],
                    "type": r["type"],
                    "resplan_label": self.room_types[r["type"]]["resplan_label"],
                }
                for r in self.rooms(req)
            ],
            "connections": self.build_edges(req),
            "assumed_fields": list(assumed),
        }

    # ========================================================
    # TTS PRE-GENERATION
    # ========================================================

    def static_tts_items(self, lang: str) -> list[tuple[str, str]]:
        """Every (id, text) pair that can be synthesized once, ever."""

        items: list[tuple[str, str]] = []

        for slot in self.slots:

            for prefix, key in (("q", "question"), ("rq", "reprompt")):

                if slot.get(key):

                    text = self.text(slot[key], lang)

                    if "{" not in text:
                        items.append((f"{prefix}__{slot['id']}__{lang}", text))

        for key in ("greeting", "joiner", "completion", "cannot_continue"):

            if key in self.messages:
                items.append((f"msg__{key}__{lang}", self.text(self.messages[key], lang)))

        for rule in self.validation_rules:

            text = self.text(rule["message"], lang)

            if "{" not in text:
                items.append((f"val__{rule['id']}__{lang}", text))

        for intent in self.intents:
            items.append((f"intent__{intent['id']}__{lang}", self.text(intent["text"], lang)))

        return items