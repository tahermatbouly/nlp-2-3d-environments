import httpx
import json
import re
from .schema import ApartmentState
from .prompts import EXTRACTION_PROMPT_TEMPLATE

OLLAMA_URL = "http://localhost:11434/api/generate"
# The user can override this if needed
MODEL_NAME = "gemma4:e4b"

from typing import Tuple, Optional

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
    
    # Use a generous read timeout for slow hardware running large models.
    # connect/write/pool stay at 30s; read (waiting for Ollama to finish) gets 10 min.
    timeout = httpx.Timeout(30.0, read=600.0)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(OLLAMA_URL, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }, timeout=timeout)
        
        response.raise_for_status()
        result_json = response.json()["response"]
        
        cleaned = _clean_llm_response(result_json)
        parsed_data = json.loads(cleaned)
        parsed_data = _normalize_parsed_data(parsed_data)
        return ApartmentState(**parsed_data), None
    except httpx.ReadTimeout:
        error_msg = "Ollama timed out — the model is taking too long. Try a shorter prompt or a smaller model."
        print(error_msg)
        return current_state, error_msg
    except httpx.ConnectError:
        error_msg = "Cannot connect to Ollama. Make sure it is running (ollama serve)."
        print(error_msg)
        return current_state, error_msg
    except httpx.HTTPStatusError as e:
        error_msg = f"Ollama returned an error: {e.response.status_code}"
        print(error_msg)
        return current_state, error_msg
    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"Failed to parse LLM response. Error: {e}. Raw output: '{result_json}'"
        print(error_msg)
        return current_state, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(error_msg)
        return current_state, error_msg
