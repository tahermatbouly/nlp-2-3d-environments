"""
Central settings for the backend.

Put this file at  Backend/config.py

Values come from environment variables. For local work, put them in a
`.env` file in the project root (next to the `Backend/` folder), for
example:

    GROQ_API_KEY=gsk_...

Install the optional loader with:  pip install python-dotenv
Never commit the .env file.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    # python-dotenv is optional: real environment variables still work.
    pass


BASE_DIR = Path(__file__).resolve().parent


def _require(name: str) -> str:
    """A setting the app cannot run without. Fails early and clearly."""

    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing required setting {name}. Set it as an environment "
            "variable or add it to the .env file in the project root."
        )

    return value


# ============================================================
# LLM (Groq)
# ============================================================

GROQ_API_KEY = _require("GROQ_API_KEY")

# Model used for extraction and reply classification.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# ============================================================
# Agent
# ============================================================

# Requirements questions, messages and validation rules.
QUESTIONS_PATH = Path(
    os.getenv("QUESTIONS_PATH", BASE_DIR / "agent" / "requirements_questions.json")
)

# "ar_eg" (Egyptian Arabic) or "en".
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ar_eg")


# ============================================================
# Storage
# ============================================================

CHAT_DB_PATH = Path(os.getenv("CHAT_DB_PATH", BASE_DIR / "data" / "chat.db"))
CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Speech (not used yet)
# ============================================================
# Left optional on purpose: the text chat runs without them. When
# voice is added, turn these into _require(...) for the models
# you pick.

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")