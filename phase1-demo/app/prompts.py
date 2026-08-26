EXTRACTION_PROMPT_TEMPLATE = """You are an apartment requirements extractor and spatial planner.

Current state:
{current_state}

User:
"{user_input}"

Update the state using information provided by the user.

Each room in the "rooms" array MUST have exactly these fields:
- "id": unique string identifier (e.g. "bedroom_1")
- "type": room type (e.g. "bedroom", "bathroom", "kitchen", "living_room")
- "count": integer, default 1
- "size": size description string or null (e.g. "large", "small")
- "connections": array of room IDs this room connects to (MUST NOT be empty)

The "requirements" object MUST have exactly these fields:
- "bedrooms": integer count
- "bathrooms": integer count
- "kitchen": integer count
- "living_room": integer count

IMPORTANT — connections rules:
- Every room MUST connect to at least one other room.
- Infer logical spatial connections based on typical apartment layouts:
  - Bedrooms connect to a bathroom and/or a hallway/living room.
  - The kitchen connects to the living room.
  - Bathrooms connect to bedrooms or hallways.
  - The living room is the central hub connecting to most rooms.
- Connections are bidirectional: if bedroom_1 connects to bathroom_1, then bathroom_1 must also list bedroom_1.
- If the user specifies explicit connections, use those instead.

Return valid JSON only, exactly matching this structure.
Do not generate explanations, markdown formatting, or any extra text. Just the JSON object.
"""
