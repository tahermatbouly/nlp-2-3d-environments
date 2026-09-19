"""
Quick manual test of the Groq-backed extraction step.

Requires a live GROQ_API_KEY (export GROQ_API_KEY=your_key_here) --
unlike test_graph.py, this one genuinely needs network access and an
API key, since it exercises the actual LLM call.

Run with: python test_llm_client.py
"""

import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.schema import ApartmentState
from app.llm_client import extract_requirements


async def main():
    state = ApartmentState()
    print("Initial state:")
    print(state.model_dump_json(indent=2))
    print("\nSending request...")

    # extract_requirements returns (state, error_msg) -- both must be
    # unpacked, or this crashes with "too many values to unpack".
    new_state, error_msg = await extract_requirements(
        "I want a bedroom and a bathroom", state
    )

    if error_msg:
        print(f"\nError: {error_msg}")
    else:
        print("\nNew state:")
        print(new_state.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())