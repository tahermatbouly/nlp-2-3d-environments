import json
import os
import re
from typing import Tuple, Optional

from groq import AsyncGroq, APIConnectionError, APITimeoutError, APIStatusError, RateLimitError

from .schema import ApartmentState
from .prompts import EXTRACTION_PROMPT_TEMPLATE

# Groq's free-tier hosted Qwen3 model -- see console.groq.com/docs/models
# for the current catalog if this ever needs to change.
# NOTE: qwen/qwen3-32b was deprecated by Groq on 2026-07-17; qwen3.6-27b is
# their recommended direct replacement.
MODEL_NAME = "openai/gpt-oss-20b"

_client: Optional[AsyncGroq] = None


def _get_client() -> AsyncGroq:
    """Lazily construct the client so a missing API key only fails when the
    LLM is actually invoked, not at import time (which would break every
    other module that imports from this file, including graph.py's tests)."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com "
                "and set it with: export GROQ_API_KEY=your_key_here"
            )
        _client = AsyncGroq(api_key=api_key)
    return _client


def _clean_llm_response(raw: str) -> str:
    """Clean LLM response by stripping think tags, markdown fences, and extracting JSON."""
    cleaned = raw.strip()

    # Strip Qwen3-style <think>...</think> reasoning blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

    # Strip markdown code block wrappers
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # If there's still extra text around the JSON, extract the outermost { }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace:last_brace + 1]

    return cleaned


def _normalize_parsed_data(data: dict) -> dict:
    """Fix common LLM field-name mistakes so the data matches the Pydantic schema."""
    # Field name aliases the LLM might use instead of "type"
    TYPE_ALIASES = {"description", "name", "room_type", "label", "room_name", "category"}

    if "rooms" in data and isinstance(data["rooms"], list):
        for room in data["rooms"]:
            if not isinstance(room, dict):
                continue

            # Map aliased field names to "type"
            if "type" not in room:
                for alias in TYPE_ALIASES:
                    if alias in room:
                        room["type"] = room.pop(alias)
                        break

            # Last resort: infer type from id (e.g. "bedroom_1" -> "bedroom")
            if "type" not in room and "id" in room:
                room["type"] = room["id"].rsplit("_", 1)[0]

            # Ensure optional fields have defaults
            room.setdefault("count", 1)
            room.setdefault("size", None)
            room.setdefault("connections", [])

    return data


async def extract_requirements(user_input: str, current_state: ApartmentState) -> Tuple[ApartmentState, Optional[str]]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        current_state=current_state.model_dump_json(indent=2),
        user_input=user_input
    )

    result_text = None
    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # low temperature: consistent structured extraction, not creativity
        )
        result_text = response.choices[0].message.content

        cleaned = _clean_llm_response(result_text)
        parsed_data = json.loads(cleaned)
        parsed_data = _normalize_parsed_data(parsed_data)
        return ApartmentState(**parsed_data), None
    except RuntimeError as e:
        # raised by _get_client() when GROQ_API_KEY is missing
        error_msg = str(e)
        print(error_msg)
        return current_state, error_msg
    except APITimeoutError:
        error_msg = "Groq request timed out. Try again."
        print(error_msg)
        return current_state, error_msg
    except APIConnectionError:
        error_msg = "Cannot connect to Groq's API. Check your internet connection."
        print(error_msg)
        return current_state, error_msg
    except RateLimitError:
        error_msg = "Groq rate limit hit. Wait a moment and try again."
        print(error_msg)
        return current_state, error_msg
    except APIStatusError as e:
        error_msg = f"Groq returned an error: {e.status_code} - {e.message}"
        print(error_msg)
        return current_state, error_msg
    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"Failed to parse LLM response. Error: {e}. Raw output: '{result_text}'"
        print(error_msg)
        return current_state, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(error_msg)
        return current_state, error_msg