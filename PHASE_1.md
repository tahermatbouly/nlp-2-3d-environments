# Phase 1 — Text-to-Bubble-Diagram Demo

## Goal

Build a text-only working demo of Phase 1:

**Natural-language text → local Ollama LLM → structured apartment JSON → deterministic Python → bubble constraint graph**

The original Phase 1 ends after the completed JSON is deterministically converted into the bubble constraint-graph required by the next phase. Speech-to-Text and Text-to-Speech are excluded from this demo. 

## 1. Define the Phase-1 JSON Schema

Create a strict, predetermined schema containing the apartment requirements.

Example:

```json
{
  "apartment": {
    "rooms": [
      {
        "id": "living_room",
        "type": "living_room",
        "count": 1,
        "size": "large",
        "connections": ["kitchen", "corridor"]
      }
    ],
    "requirements": {
      "bedrooms": 2,
      "bathrooms": 1,
      "kitchen": 1
    }
  }
}
```

The schema should match the project's predetermined questions/features.

## 2. Build the Ollama Backend

Use **Python + FastAPI**.

```text
POST /chat
    ↓
Ollama
    ↓
Updated JSON state
```

Use a small local model through Ollama.

The LLM's responsibility is **information extraction only**. It should not generate geometry.

## 3. Create a Fixed Extraction Prompt

The prompt should provide:

- Current JSON state
- Allowed fields
- Allowed values
- User's text
- Instructions to return JSON only
- Instructions not to invent missing information

Example:

```text
You are an apartment requirements extractor.

Current state:
{...}

User:
"I want two bedrooms, one bathroom and a large living room."

Update the state using only information explicitly provided.

Return valid JSON only.
Do not generate explanations.
```

## 4. Add State Management

For the demo, keep the conversation state in memory.

```python
conversation = {
    "messages": [...],
    "apartment_state": {...}
}
```

Each user message updates the apartment state.

No database is required for the first demo.

## 5. Add Completion Checking

The backend should deterministically check whether all required fields are filled.

Example:

```python
required = [
    "bedrooms",
    "bathrooms",
    "kitchen",
    "living_room"
]
```

The LLM should not decide whether the interview is complete.

## 6. Convert JSON → Bubble Constraint Graph

This is the important Phase-1 boundary.

Example:

```text
living_room ─── kitchen
     │
  corridor
   /     \
bedroom  bedroom
     │
  bathroom
```

Represent it internally as:

```python
graph = {
    "nodes": [...],
    "edges": [...]
}
```

Where:

- Each **node** represents a room.
- Each **edge** represents a required adjacency/connection.

This conversion must be deterministic.

## 7. Render the Bubble Diagram

For the demo, use:

- **NetworkX** — graph representation/layout
- **Matplotlib** — visualization

HouseDiffusion is not required in Phase 1.

## 8. Demo Flow

```text
User types:
"I need an apartment for a family of four.
Two bedrooms, one bathroom, a kitchen and a large
living room. The kitchen should connect to the living room."

        ↓

      Ollama
        ↓
 Structured JSON
        ↓
    Validation
        ↓
Deterministic graph generator
        ↓
   Bubble Diagram
```

## 9. Project Structure

```text
phase1-demo/
├── app/
│   ├── main.py
│   ├── ollama.py
│   ├── prompts.py
│   ├── state.py
│   ├── validator.py
│   ├── graph.py
│   └── schema.py
│
├── frontend/
│   └── index.html
│
├── examples/
│   └── sample_requests.txt
│
└── requirements.txt
```

## 10. Technology Stack

```text
Python
├── FastAPI
├── Pydantic
├── Ollama
├── NetworkX
└── Matplotlib
```

## Implementation Principle

Keep the boundary strict:

**Ollama = semantic understanding/extraction**

**Python = validation, state management, graph construction, and visualization**

The LLM should never directly determine geometry.
