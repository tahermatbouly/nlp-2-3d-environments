import torch
from silero_vad import load_silero_vad, VADIterator


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 900,
        speech_pad_ms: int = 200,
    ):
        self.sample_rate = sample_rate

        print("[VAD] Loading Silero VAD...")

        self.model = load_silero_vad()

        self.iterator = VADIterator(
            self.model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        # Silero requires exactly 512 samples at 16 kHz.
        self.frame_size = 512

        # Incoming WebSocket chunks don't necessarily
        # contain exactly 512 samples.
        self.audio_buffer = torch.empty(
            0,
            dtype=torch.float32,
        )

        print("[VAD] Ready")

    def process(self, audio_chunk: bytes) -> list[str]:
        """
        Process PCM16 audio.

        The incoming chunk can have any number of samples.
        Internally we split it into 512-sample frames.

        Returns:
            A list containing zero or more events:
                ["speech_start"]
                ["speech_end"]
                ["speech_start", "speech_end"]
        """

        if not audio_chunk:
            return []

        # Make a writable copy to avoid the PyTorch warning.
        audio = torch.frombuffer(
            bytearray(audio_chunk),
            dtype=torch.int16,
        ).float()

        audio = audio / 32768.0

        # Add incoming samples to our internal frame buffer.
        self.audio_buffer = torch.cat(
            (
                self.audio_buffer,
                audio,
            )
        )

        events = []

        while self.audio_buffer.numel() >= self.frame_size:

            frame = self.audio_buffer[:self.frame_size]

            self.audio_buffer = self.audio_buffer[
                self.frame_size:
            ]

            event = self.iterator(frame)

            if event is None:
                continue

            if "start" in event:
                events.append("speech_start")

            elif "end" in event:
                events.append("speech_end")

        return events

    def reset(self):
        self.iterator.reset_states()

        self.audio_buffer = torch.empty(
            0,
            dtype=torch.float32,
        )