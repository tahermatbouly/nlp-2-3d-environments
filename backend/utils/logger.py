
import logging
from pathlib import Path


LOG_DIR = Path("Backend/data/logs")
CONVERSATIONS_DIR = LOG_DIR / "conversations"

LOG_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Main application logger
# ---------------------------------------------------------

logger = logging.getLogger("voice_agent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_DIR / "app.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ---------------------------------------------------------
# Conversation logger
# ---------------------------------------------------------

def get_conversation_logger(call_id: str) -> logging.Logger:
    """
    Creates/returns a logger dedicated to one conversation.
    """

    conversation_logger = logging.getLogger(
        f"voice_agent.conversation.{call_id}"
    )

    conversation_logger.setLevel(logging.INFO)
    conversation_logger.propagate = False

    if not conversation_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%H:%M:%S",
        )

        file_handler = logging.FileHandler(
            CONVERSATIONS_DIR / f"{call_id}.log",
            encoding="utf-8",
        )

        file_handler.setFormatter(formatter)
        conversation_logger.addHandler(file_handler)

    return conversation_logger
