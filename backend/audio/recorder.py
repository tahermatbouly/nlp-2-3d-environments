import asyncio
import json
import wave
from pathlib import Path


RECORDINGS_DIR = Path("Backend/data/recordings")


def _write_wav(
    path: Path,
    audio: bytes,
    sample_rate: int,
) -> None:
    """
    Blocking WAV write — always called via asyncio.to_thread().
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


def _append_manifest(
    manifest_path: Path,
    entry: dict,
) -> None:
    """
    Blocking manifest update — always called via
    asyncio.to_thread().

    Rewritten in full each time rather than appended to, so the
    file is always valid JSON even if the call drops mid-interview.
    The manifest is small (one entry per answer), so the rewrite
    cost is irrelevant.
    """

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    entries = []

    if manifest_path.exists():

        try:

            with open(
                manifest_path,
                "r",
                encoding="utf-8",
            ) as f:
                entries = json.load(f)

        except Exception:
            # A corrupt/partial manifest must never break the
            # actual interview — start a fresh list instead.
            entries = []

    entries.append(entry)

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            entries,
            f,
            ensure_ascii=False,
            indent=2,
        )


class AnswerRecorder:
    """
    Saves one WAV file per candidate answer, plus a manifest
    mapping each file to the question it answered and what the STT
    produced for it.

    One instance per call/session. Files land in:

        Backend/data/recordings/<session_id>/
            001_q1_name.wav
            002_q2_domain.wav
            003_q2_domain.wav      <- a repeat of the same question
            manifest.json

    The leading counter is the turn number, so repeated attempts at
    the same question sort in the order they actually happened and
    never overwrite each other.
    """

    def __init__(
        self,
        session_id: str,
    ):
        self.session_id = session_id

        self.session_dir = (
            RECORDINGS_DIR / session_id
        )

        self.manifest_path = (
            self.session_dir / "manifest.json"
        )

        self._turn = 0

    async def save(
        self,
        audio: bytes,
        sample_rate: int,
        question_id: str,
        transcript: str = "",
    ) -> str | None:
        """
        Save one answer's audio. Returns the path written, or None
        if there was nothing to save.

        Never raises: a recording failure must not interrupt a live
        interview, so problems are swallowed and reported via the
        return value instead.
        """

        if not audio:
            return None

        self._turn += 1

        # Keep the filename filesystem-safe — question ids are
        # ours, but the summary/system ids could change later.
        safe_question_id = "".join(
            ch
            if ch.isalnum() or ch in ("_", "-")
            else "_"
            for ch in (question_id or "unknown")
        )

        filename = (
            f"{self._turn:03d}_{safe_question_id}.wav"
        )

        path = self.session_dir / filename

        try:

            await asyncio.to_thread(
                _write_wav,
                path,
                audio,
                sample_rate,
            )

            duration_s = len(audio) / (
                sample_rate * 2
            )

            await asyncio.to_thread(
                _append_manifest,
                self.manifest_path,
                {
                    "turn": self._turn,
                    "question_id": question_id,
                    "file": filename,
                    "duration_seconds": round(
                        duration_s,
                        2,
                    ),
                    "sample_rate": sample_rate,
                    "transcript": transcript,
                },
            )

            return str(path)

        except Exception:
            return None