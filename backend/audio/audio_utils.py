# understands the different audio types and formats

import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def pcm16_to_float32(audio_bytes: bytes) -> torch.Tensor:
    """
    Convert PCM16 bytes to float32 audio in [-1, 1].
    """

    audio = torch.frombuffer(
        bytearray(audio_bytes),
        dtype=torch.int16,
    ).float()

    return audio / 32768.0


def float32_to_pcm16(audio: torch.Tensor) -> bytes:
    """
    Convert float32 audio in [-1, 1] to PCM16 bytes.
    """

    audio = torch.clamp(audio, -1.0, 1.0)

    audio = (audio * 32767.0).to(torch.int16)

    return audio.cpu().numpy().tobytes()


def resample_audio(
    audio: torch.Tensor,
    source_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Resample mono float32 audio.

    This is a one-shot resample — correct for resampling a complete,
    already-assembled buffer (e.g. a finished utterance) in one call.
    Do NOT use this per-chunk on a live stream: it zero-pads at both
    edges of whatever you hand it, so calling it on every ~100ms
    WebSocket chunk introduces a discontinuity at every chunk
    boundary. Use StreamResampler for that instead.
    """

    if source_rate == target_rate:
        return audio

    return torchaudio.functional.resample(
        audio,
        source_rate,
        target_rate,
    )


class StreamResampler:
    """
    Resamples a continuous stream of chunks without introducing a
    discontinuity at every chunk boundary.

    torchaudio.functional.resample() is stateless and implicitly
    zero-pads at the edges of whatever buffer you give it. Calling it
    fresh on every incoming WebSocket chunk means every chunk
    boundary gets a small filter-ringing artifact (effectively a
    click), which — concatenated across a whole utterance — shows up
    as periodic noise threaded through the signal. That's enough to
    push an STT model into hallucinating and can cause a VAD to fire
    spurious start/end events on the transients.

    This keeps a short tail of raw (pre-resample) samples from the
    previous chunk as context for the next resample call, then
    discards the output samples that came from that borrowed context
    so nothing is duplicated.

    One instance per connection/session — persists for the life of
    the session, independent of VAD/speech state. Do not reset it on
    speech_start/speech_end; it needs to track the raw input stream
    continuously. Only reset it if you tear down and rebuild the
    whole session.
    """

    def __init__(
        self,
        source_rate: int,
        target_rate: int = TARGET_SAMPLE_RATE,
        context_samples: int = 64,
    ):
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.context_samples = context_samples
        self._tail = torch.zeros(context_samples, dtype=torch.float32)

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        if self.source_rate == self.target_rate:
            return audio

        combined = torch.cat((self._tail, audio))

        resampled = torchaudio.functional.resample(
            combined,
            self.source_rate,
            self.target_rate,
        )

        ratio = self.target_rate / self.source_rate
        discard = int(round(self.context_samples * ratio))

        # Save this chunk's raw (un-resampled) tail for next call —
        # not the combined buffer, to avoid growing unbounded.
        if audio.numel() >= self.context_samples:
            self._tail = audio[-self.context_samples:].clone()
        else:
            # Chunk shorter than our context window: keep what's left
            # of the old tail plus everything from this chunk.
            self._tail = torch.cat(
                (self._tail[audio.numel():], audio)
            )

        return resampled[discard:]

    def reset(self):
        self._tail = torch.zeros(self.context_samples, dtype=torch.float32)