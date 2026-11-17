import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    meta        TEXT,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


class SessionStore:
    """
    Keeps two things per chat:

      - the agent's AgentState (a plain JSON-serializable dict), so a
        conversation survives a server restart and every input mode
        (typing now, dictation and voice mode later) shares it
      - the visible message history, so the page can restore the chat
        after a reload

    sqlite3 calls are short and local, so they are called directly
    from async endpoints. If this ever sits behind many users, move
    them to a thread pool or switch to Postgres.
    """

    def __init__(self, path: str = "chat.db"):

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()

        with self._lock, self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)

    # --------------------------------------------------------
    # Sessions
    # --------------------------------------------------------

    def create(self, state: dict) -> str:

        session_id = uuid.uuid4().hex
        now = time.time()

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, state, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, json.dumps(state, ensure_ascii=False), now, now),
            )

        return session_id

    def load_state(self, session_id: str) -> dict | None:

        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

        return json.loads(row["state"]) if row else None

    def save_state(self, session_id: str, state: dict) -> None:

        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?",
                (json.dumps(state, ensure_ascii=False), time.time(), session_id),
            )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    def add_message(
        self, session_id: str, role: str, text: str, meta: dict | None = None
    ) -> None:

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, text, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    text,
                    json.dumps(meta, ensure_ascii=False) if meta else None,
                    time.time(),
                ),
            )

    def history(self, session_id: str) -> list[dict]:

        with self._lock:
            rows = self._conn.execute(
                "SELECT role, text, meta FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "text": row["text"],
                **(json.loads(row["meta"]) if row["meta"] else {}),
            }
            for row in rows
        ]
