import asyncio
from phase1_demo.app.schema import ApartmentState
from phase1_demo.app.ollama import extract_requirements

async def main():
    state = ApartmentState()
    print("Initial state:")
    print(state.model_dump_json(indent=2))
    print("\nSending request...")
    new_state = await extract_requirements("I want a bedroom and a bathroom", state)
    print("\nNew state:")
    print(new_state.model_dump_json(indent=2))

asyncio.run(main())
