"""
Voice mode (the "wave icon") for the floor plan chat.

Put this file at  backend/websocket/handler.py

It is the old interview WebSocket handler, adapted to the new agent:

  - It attaches to an EXISTING chat session, so voice and text share
    one conversation. Every spoken turn is also saved to the chat
    history, so the transcript is there when voice mode closes.
  - Same audio pipeline as before: 24 kHz PCM16 in -> resample -> VAD
    -> STT -> graph -> TTS out, with the mic gated while the agent
    speaks and the same playback_done handshake.
  - Silence only ends VOICE MODE. The conversation itself continues
    in the chat.

Wire it up in backend/main.py (after build_reply is defined):

    from backend.websocket.handler import make_voice_endpoint

    app.add_api_websocket_route(
        "/sessions/{session_id}/voice",
        make_voice_endpoint(store, lambda sid: _locks[sid], build_reply),
    )

Messages the server sends (JSON unless noted):

    voice_ready          {language, mode}            connected, session loaded
    turn                 {mode, slots}               an agent turn is starting
    tts_start            {sample_rate, channels}     then binary PCM16 chunks
    tts_end                                          end of one audio clip
    turn_playback_end                                every clip of the turn sent
    speech_started / speech_ended {bytes, sample_rate}
    transcript           {text}                      what STT heard
    assistant            {text, quick_replies, done} the reply as chat text
    conversation_complete {room_graph, ended_early}  finished, socket closes
    voice_ended          {reason}                    silence, socket closes
    error                {stage, message}

Messages the client sends:

    binary               PCM16 mono audio at 24 kHz
    {"type": "playback_done"}   client finished playing the turn
    {"type": "end"}             user closed voice mode
"""

import asyncio
import hashlib
import json
import logging
import time
import wave
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect

from backend.agent.graph import agent_graph, manager
from backend.audio.audio_utils import (
    TARGET_SAMPLE_RATE,
    float32_to_pcm16,
    pcm16_to_float32,
    StreamResampler,
)
from backend.audio.buffer import AudioBuffer
from backend.audio.recorder import AnswerRecorder
from backend.audio.vad import VoiceActivityDetector
from backend.stt_plugin import CohereArabicSTT
from backend.tts_plugin import VoiceTutTTS


logger = logging.getLogger("websocket")


INPUT_SAMPLE_RATE = 24000

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TTS_CACHE_DIR = DATA_DIR / "tts_cache"
ROOM_GRAPH_DIR = DATA_DIR / "room_graphs"

TTS_SEND_CHUNK_SIZE = 32 * 1024

# Extra mute after estimated playback so silence is measured only once
# the client has finished playing (clients buffer until tts_end).
TTS_TAIL_SECONDS = 0.4

DEFAULT_SILENCE_MS = 700

# Seconds of real silence before each nudge. After the last prompt
# (end_voice) voice mode closes and the chat carries on.
SILENCE_STEP_SECONDS = 5.0

SILENCE_PROMPTS = [
    {
        "id": "silence__1",
        "text": {"en": "Are you still there?", "ar_eg": "لسه معايا؟"},
        "end_voice": False,
    },
    {
        "id": "silence__2",
        "text": {
            "en": "I'm waiting for your answer. You can also type it in the chat.",
            "ar_eg": "مستنيك تجاوب. تقدر كمان تكتب إجابتك في الشات.",
        },
        "end_voice": False,
    },
    {
        "id": "silence__3",
        "text": {
            "en": "I'll turn voice mode off now. We can carry on in the chat.",
            "ar_eg": "هقفل وضع الصوت دلوقتي. نكمل في الشات.",
        },
        "end_voice": True,
    },
]


# ============================================================
# TTS CACHE
# ============================================================
# Clips are cached on disk. The file name includes a short hash of
# the text, so editing a question in questions.json produces a new
# clip instead of replaying the stale one.

# Synthesis already running for a clip, shared by concurrent callers.
_inflight: dict[str, asyncio.Task] = {}

# (speaker, language) pairs whose cache warm-up has been started.
_warmed: set[str] = set()


def _localized(obj: dict, lang: str) -> str:

    return obj.get(lang) or obj.get("en") or ""


def _cache_path(item_id: str, text: str, speaker: str) -> Path:

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    safe_speaker = speaker.replace("/", "_")

    return TTS_CACHE_DIR / f"{item_id}__{digest}__{safe_speaker}.wav"


def _read_cached_wav(path: Path) -> tuple[bytes, int]:

    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())

    return pcm, sample_rate


def _write_cached_wav(path: Path, audio: bytes, sample_rate: int) -> None:

    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio)


async def _synthesize_and_store(
    tts: VoiceTutTTS, path: Path, item_id: str, text: str
) -> tuple[bytes, int]:

    logger.info("[TTS] Cache miss, generating | id=%s", item_id)

    audio, sample_rate = await tts.synthesize(text)

    if not audio or sample_rate <= 0:
        raise RuntimeError(f"TTS returned no audio for {item_id}")

    await asyncio.to_thread(_write_cached_wav, path, audio, sample_rate)

    return audio, sample_rate


async def _get_or_synthesize(
    tts: VoiceTutTTS, item_id: str, text: str
) -> tuple[bytes, int]:

    path = _cache_path(item_id, text, tts.speaker)

    if await asyncio.to_thread(path.exists):
        return await asyncio.to_thread(_read_cached_wav, path)

    key = str(path)
    task = _inflight.get(key)

    if task is None:
        task = asyncio.create_task(_synthesize_and_store(tts, path, item_id, text))
        _inflight[key] = task
        task.add_done_callback(lambda _t, k=key: _inflight.pop(k, None))

    # shield: one client disconnecting must not cancel a clip that
    # another connection (or the warm-up) is also waiting for.
    return await asyncio.shield(task)


def _static_items(lang: str) -> list[tuple[str, str]]:
    """Every clip that never changes: questions, messages, silence prompts."""

    items = list(manager.static_tts_items(lang))

    for prompt in SILENCE_PROMPTS:
        items.append((f"{prompt['id']}__{lang}", _localized(prompt["text"], lang)))

    return items


async def _warm_cache(tts: VoiceTutTTS, lang: str) -> None:
    """
    Generate every static clip once, in the background, one at a time.
    The first connection does not wait for this: a clip that is not
    ready yet is simply generated on demand.
    """

    key = f"{tts.speaker}:{lang}"

    if key in _warmed:
        return

    _warmed.add(key)

    try:
        for item_id, text in _static_items(lang):
            await _get_or_synthesize(tts, item_id, text)

        logger.info("[TTS] Static clips ready | %s", key)

    except asyncio.CancelledError:
        _warmed.discard(key)
        raise

    except Exception:
        _warmed.discard(key)
        logger.exception("[TTS] Cache warm-up failed | %s", key)


# ============================================================
# RESULT STORAGE
# ============================================================

def _write_room_graph(room_graph: dict, session_id: str) -> str:

    ROOM_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    path = ROOM_GRAPH_DIR / f"{session_id}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(room_graph, f, ensure_ascii=False, indent=2)

    return str(path)


async def _safe_send(websocket: WebSocket, payload: dict) -> None:

    try:
        await websocket.send_json(payload)
    except Exception:
        pass


# ============================================================
# ENDPOINT FACTORY
# ============================================================

def make_voice_endpoint(store, lock_for, build_reply):
    """
    store      SessionStore from main.py
    lock_for   session_id -> asyncio.Lock, the SAME lock the text
               endpoint uses, so a typed message and a spoken one can
               never run two graph calls on one state
    build_reply  main.py's build_reply(state) -> dict
    """

    async def voice_endpoint(websocket: WebSocket, session_id: str):

        await websocket.accept()

        logger.info("[WS] Voice client connected | session=%s", session_id)

        current_state = store.load_state(session_id)

        if current_state is None:
            await _safe_send(
                websocket,
                {"type": "error", "stage": "session", "message": "Session not found."},
            )
            await websocket.close(code=4404)
            return

        if current_state["conversation_finished"]:
            await _safe_send(
                websocket,
                {
                    "type": "conversation_complete",
                    "room_graph": current_state["room_graph"],
                    "ended_early": current_state["ended_early"],
                },
            )
            await websocket.close()
            return

        # ====================================================
        # SESSION COMPONENTS
        # ====================================================

        audio_buffer = AudioBuffer()
        recorder = AnswerRecorder(session_id)
        vad = VoiceActivityDetector(min_silence_duration_ms=DEFAULT_SILENCE_MS)
        stt = CohereArabicSTT()
        tts = VoiceTutTTS()

        resampler = StreamResampler(
            source_rate=INPUT_SAMPLE_RATE,
            target_rate=TARGET_SAMPLE_RATE,
        )

        speech_active = False
        mic_muted_until = 0.0

        # Wall-clock estimate of when sequential client playback ends.
        # Fallback if the client never sends playback_done.
        playback_ends_at = 0.0

        # True from the start of a spoken turn until the client confirms
        # it played every clip. While set, the mic stays gated and
        # silence must not escalate.
        awaiting_playback_ack = False

        playback_fallback_task = None
        silence_task = None
        silence_level = 0

        warm_task = asyncio.create_task(_warm_cache(tts, current_state["language"]))

        # ====================================================
        # SEGMENT AUDIO
        # ====================================================

        async def _resolve_segment_audio(seg: dict):
            """Load or synthesize one segment. Returns (audio, rate, id) or None."""

            text = seg.get("text")

            if not text:
                return None

            if seg.get("kind", "static") == "static":

                item_id = seg["id"]
                audio, sample_rate = await _get_or_synthesize(tts, item_id, text)

            else:
                # "live": text that depends on the answers (the summary,
                # the room list). Synthesized fresh, never cached.
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                item_id = f"live__{digest}"

                logger.info("[TTS] Live synthesize | %s", item_id)

                audio, sample_rate = await tts.synthesize(text)

            if not audio or sample_rate <= 0:
                logger.warning("[TTS] Empty audio | %s", item_id)
                return None

            return audio, sample_rate, item_id

        async def _send_pcm(audio: bytes, sample_rate: int, item_id: str):
            """Stream one PCM clip to the client and update the mute timing."""

            nonlocal speech_active, mic_muted_until, playback_ends_at

            mic_muted_until = float("inf")
            speech_active = False
            audio_buffer.clear()
            vad.reset()

            duration_s = len(audio) / (sample_rate * 2)

            logger.info(
                "[TTS] Playing | id=%s | duration=%.2fs", item_id, duration_s
            )

            await websocket.send_json(
                {"type": "tts_start", "sample_rate": sample_rate, "channels": 1}
            )

            for offset in range(0, len(audio), TTS_SEND_CHUNK_SIZE):
                await websocket.send_bytes(audio[offset : offset + TTS_SEND_CHUNK_SIZE])

            await websocket.send_json({"type": "tts_end"})

            now = time.monotonic()
            playback_ends_at = max(now, playback_ends_at) + duration_s + TTS_TAIL_SECONDS

            mic_muted_until = (
                float("inf") if awaiting_playback_ack else playback_ends_at
            )

        async def speak_segments(segments: list[dict], pre_resolved: list | None = None):
            """
            Resolve every segment in parallel, stitch the PCM into one
            clip and send it once (no gaps between the pieces).
            """

            nonlocal mic_muted_until

            if not segments:
                return

            try:
                resolved = (
                    pre_resolved
                    if pre_resolved is not None
                    else await asyncio.gather(
                        *[_resolve_segment_audio(seg) for seg in segments]
                    )
                )

                parts = [item for item in resolved if item is not None]

                if not parts:
                    return

                rates = {rate for _, rate, _ in parts}

                if len(rates) != 1:
                    logger.warning("[TTS] Mixed sample rates %s, sending separately", rates)

                    for audio, sample_rate, item_id in parts:
                        await _send_pcm(audio, sample_rate, item_id)

                    return

                await _send_pcm(
                    b"".join(audio for audio, _, _ in parts),
                    parts[0][1],
                    "+".join(item_id for _, _, item_id in parts),
                )

            except WebSocketDisconnect:
                raise

            except Exception as e:

                if not awaiting_playback_ack:
                    mic_muted_until = max(playback_ends_at, time.monotonic())

                logger.exception("[TTS] Failed to speak")

                await _safe_send(
                    websocket, {"type": "error", "stage": "tts", "message": str(e)}
                )

        # ====================================================
        # PLAYBACK ACK
        # ====================================================

        async def _cancel_playback_fallback():

            nonlocal playback_fallback_task

            if playback_fallback_task is None:
                return

            playback_fallback_task.cancel()

            try:
                await playback_fallback_task
            except asyncio.CancelledError:
                pass

            playback_fallback_task = None

        async def _playback_ack_fallback():
            """If the client never sends playback_done, use the duration
            estimate so the conversation cannot soft-lock."""

            nonlocal awaiting_playback_ack, mic_muted_until

            while time.monotonic() < playback_ends_at:

                if not awaiting_playback_ack:
                    return

                await asyncio.sleep(0.2)

            await asyncio.sleep(1.0)

            if not awaiting_playback_ack:
                return

            logger.warning("[TTS] playback_done not received, using estimate")

            awaiting_playback_ack = False
            mic_muted_until = time.monotonic() + TTS_TAIL_SECONDS

        async def mark_playback_done():

            nonlocal awaiting_playback_ack, mic_muted_until

            awaiting_playback_ack = False
            mic_muted_until = time.monotonic() + TTS_TAIL_SECONDS

            await _cancel_playback_fallback()

        def bot_is_speaking() -> bool:

            if awaiting_playback_ack or mic_muted_until == float("inf"):
                return True

            return time.monotonic() < mic_muted_until

        # ====================================================
        # SPEAK A TURN
        # ====================================================

        async def speak_turn(turn_state: dict):

            nonlocal awaiting_playback_ack, mic_muted_until, playback_fallback_task

            await stop_silence_timer()
            await _cancel_playback_fallback()

            awaiting_playback_ack = True
            mic_muted_until = float("inf")

            await websocket.send_json(
                {
                    "type": "turn",
                    "mode": turn_state.get("mode"),
                    "slots": turn_state.get("current_slots", []),
                }
            )

            intent_prompt = turn_state.get("intent_prompt") or []
            current_prompt = turn_state.get("current_prompt") or []

            # With a filler line first ("Take your time."), start
            # synthesizing the main line while the filler plays.
            main_prefetch = None

            try:
                if intent_prompt and current_prompt:
                    main_prefetch = asyncio.gather(
                        *[_resolve_segment_audio(seg) for seg in current_prompt]
                    )

                if intent_prompt:
                    await speak_segments(intent_prompt)

                if current_prompt:

                    if main_prefetch is not None:
                        await speak_segments(
                            current_prompt, pre_resolved=await main_prefetch
                        )
                        main_prefetch = None
                    else:
                        await speak_segments(current_prompt)

            finally:
                if main_prefetch is not None and not main_prefetch.done():
                    main_prefetch.cancel()

                    try:
                        await main_prefetch
                    except (asyncio.CancelledError, Exception):
                        pass

            if not intent_prompt and not current_prompt:
                awaiting_playback_ack = False
                mic_muted_until = 0.0
                return

            await websocket.send_json({"type": "turn_playback_end"})

            playback_fallback_task = asyncio.create_task(_playback_ack_fallback())

        # ====================================================
        # SILENCE
        # ====================================================

        async def stop_silence_timer():

            nonlocal silence_task

            if silence_task is not None:
                silence_task.cancel()

                try:
                    await silence_task
                except asyncio.CancelledError:
                    pass

                silence_task = None

        async def wait_until_listening():
            """Block until the agent is done speaking and the mic is live."""

            while bot_is_speaking() or speech_active:
                await asyncio.sleep(0.2)

        async def silence_monitor():

            nonlocal silence_level

            slice_s = 0.25

            while True:

                await wait_until_listening()

                # Real silence only: agent speech resets the counter.
                silent_for = 0.0

                while silent_for < SILENCE_STEP_SECONDS:

                    await asyncio.sleep(slice_s)

                    if speech_active or bot_is_speaking():
                        silent_for = 0.0
                        await wait_until_listening()
                        continue

                    silent_for += slice_s

                if speech_active or bot_is_speaking():
                    continue

                silence_level += 1

                if silence_level > len(SILENCE_PROMPTS):
                    return

                prompt = SILENCE_PROMPTS[silence_level - 1]
                lang = current_state["language"]

                logger.info("[VOICE] Silence nudge | level=%d", silence_level)

                await speak_segments(
                    [
                        {
                            "kind": "static",
                            "id": f"{prompt['id']}__{lang}",
                            "text": _localized(prompt["text"], lang),
                        }
                    ]
                )

                if prompt["end_voice"]:

                    # Ends voice mode only. The chat session stays open.
                    await _safe_send(
                        websocket, {"type": "voice_ended", "reason": "silence"}
                    )

                    # Let the client finish playing the goodbye line.
                    while time.monotonic() < playback_ends_at:
                        await asyncio.sleep(0.2)

                    await websocket.close()
                    return

        async def start_silence_timer():

            nonlocal silence_task, silence_level

            await stop_silence_timer()

            silence_level = 0
            silence_task = asyncio.create_task(silence_monitor())

        # ====================================================
        # ONE SPOKEN TURN THROUGH THE AGENT
        # ====================================================

        async def run_turn(transcript: str):
            """
            Same steps as POST /sessions/{id}/messages, under the same
            lock, reloading the state first because the user may have
            typed a message in the meantime.
            """

            async with lock_for(session_id):

                state = store.load_state(session_id)

                if state is None:
                    raise RuntimeError("Session no longer exists.")

                if state["conversation_finished"]:
                    return state, build_reply(state)

                new_state = await agent_graph.ainvoke({**state, "transcript": transcript})

                store.save_state(session_id, new_state)

                if transcript:
                    store.add_message(session_id, "user", transcript)

                reply = build_reply(new_state)

                store.add_message(
                    session_id,
                    "assistant",
                    reply["text"],
                    {"quick_replies": reply["quick_replies"], "done": reply["done"]},
                )

            return new_state, reply

        async def finish_conversation(final_state: dict):

            await stop_silence_timer()

            room_graph = final_state.get("room_graph")

            if room_graph:
                try:
                    path = await asyncio.to_thread(_write_room_graph, room_graph, session_id)
                    logger.info("[VOICE] Saved room graph: %s", path)
                except Exception:
                    logger.exception("[VOICE] Failed to save room graph")

            await _safe_send(
                websocket,
                {
                    "type": "conversation_complete",
                    "room_graph": room_graph,
                    "ended_early": final_state.get("ended_early", False),
                },
            )

        async def handle_utterance(utterance: bytes) -> bool:
            """STT -> agent -> speak. Returns True when the conversation is over."""

            nonlocal current_state

            transcript = ""

            if utterance:
                try:
                    transcript = (
                        await stt.transcribe(
                            audio_bytes=utterance, sample_rate=TARGET_SAMPLE_RATE
                        )
                    ).strip()

                except WebSocketDisconnect:
                    raise

                except Exception as e:
                    logger.exception("[STT] Transcription failed")

                    await _safe_send(
                        websocket, {"type": "error", "stage": "stt", "message": str(e)}
                    )

                    await start_silence_timer()
                    return False

            logger.info("[STT] Transcript: %s", transcript or "<EMPTY>")

            await websocket.send_json({"type": "transcript", "text": transcript})

            try:
                await recorder.save(
                    audio=utterance,
                    sample_rate=TARGET_SAMPLE_RATE,
                    question_id="+".join(current_state["current_slots"])
                    or current_state["mode"],
                    transcript=transcript,
                )
            except Exception:
                logger.exception("[RECORD] Could not save answer audio")

            try:
                new_state, reply = await run_turn(transcript)

            except WebSocketDisconnect:
                raise

            except Exception as e:
                logger.exception("[AGENT] Processing failed")

                await _safe_send(
                    websocket, {"type": "error", "stage": "agent", "message": str(e)}
                )

                await start_silence_timer()
                return False

            current_state = new_state

            await websocket.send_json(
                {
                    "type": "assistant",
                    "text": reply["text"],
                    "quick_replies": reply["quick_replies"],
                    "done": reply["done"],
                }
            )

            await speak_turn(new_state)

            if new_state["conversation_finished"]:
                await finish_conversation(new_state)
                return True

            await start_silence_timer()
            return False

        # ====================================================
        # MAIN LOOP
        # ====================================================

        try:
            await websocket.send_json(
                {
                    "type": "voice_ready",
                    "language": current_state["language"],
                    "mode": current_state["mode"],
                }
            )

            # Say what is waiting for an answer: the greeting for a new
            # chat, or the pending question when switching from text.
            await speak_turn({**current_state, "intent_prompt": []})
            await start_silence_timer()

            while True:

                message = await websocket.receive()

                # ------------------------------------------------
                # TEXT (control messages)
                # ------------------------------------------------
                if message.get("text") is not None:

                    try:
                        data = json.loads(message["text"])
                    except Exception:
                        data = None

                    if isinstance(data, dict):

                        if data.get("type") == "playback_done":
                            await mark_playback_done()

                        elif data.get("type") == "end":
                            logger.info("[WS] Client ended voice mode")
                            return

                    continue

                # ------------------------------------------------
                # AUDIO
                # ------------------------------------------------
                audio_data = message.get("bytes")

                if not audio_data:
                    continue

                # Mic is gated while the agent is speaking, including
                # a whole multi-clip turn until the client confirms.
                if bot_is_speaking():
                    continue

                audio = pcm16_to_float32(audio_data)
                audio = resampler.process(audio)
                normalized_audio = float32_to_pcm16(audio)

                vad_events = vad.process(normalized_audio)

                # Speech start must run before buffering.
                for event in vad_events:

                    if event == "speech_start":
                        speech_active = True
                        await stop_silence_timer()
                        audio_buffer.clear()
                        await websocket.send_json({"type": "speech_started"})

                if speech_active:
                    audio_buffer.add(normalized_audio)

                for event in vad_events:

                    if event != "speech_end":
                        continue

                    speech_active = False

                    utterance = audio_buffer.get_audio()

                    await websocket.send_json(
                        {
                            "type": "speech_ended",
                            "bytes": len(utterance),
                            "sample_rate": TARGET_SAMPLE_RATE,
                        }
                    )

                    if await handle_utterance(utterance):
                        return

                    audio_buffer.clear()
                    vad.reset()

        except WebSocketDisconnect:
            logger.info("[WS] Client disconnected")

        except RuntimeError as e:

            if "disconnect message" in str(e):
                logger.info("[WS] Client disconnected")
            else:
                logger.exception("[WS] Runtime error")

        except Exception:
            logger.exception("[WS] Unexpected error")

        finally:
            await stop_silence_timer()
            await _cancel_playback_fallback()

            warm_task.cancel()

            try:
                await warm_task
            except (asyncio.CancelledError, Exception):
                pass

            # Closing explicitly lets the client see the end of voice mode
            # (conversation finished, user pressed end, or silence).
            try:
                await websocket.close()
            except Exception:
                pass

            logger.info("[WS] Voice connection closed | session=%s", session_id)

    return voice_endpoint