import asyncio
from app.schema import ApartmentState
from app.ollama import extract_requirements
import json

async def main():
    state = ApartmentState()
    new_state = await extract_requirements("I want a bedroom and a bathroom", state)
    print("New state:")
    print(new_state.model_dump_json(indent=2))

asyncio.run(main())
