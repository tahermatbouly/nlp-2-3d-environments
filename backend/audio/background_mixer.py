import random
import wave
from pathlib import Path

import numpy as np


class BackgroundMixer:
    """
    Continuous background-audio generator.

    Generates:
        - call-center ambience
        - occasional phone ringing

    FORMAT:
        - 24,000 Hz
        - mono
        - signed 16-bit PCM

    This class does NOT mix TTS anymore.

    It only generates the continuous background layer.
    The handler is responsible for mixing TTS into each
    generated background frame.
    """

    SAMPLE_RATE = 24_000
    SAMPLE_WIDTH = 2  # int16 = 2 bytes

    def __init__(
        self,
        ambience_path: str,
        ring_path: str,
        ambience_volume: float = 0.35,
        ring_volume: float = 0.35,
        ring_min_interval: float = 18.0,
        ring_max_interval: float = 40.0,
    ):
        self.ambience_path = Path(ambience_path)
        self.ring_path = Path(ring_path)

        self.ambience_volume = ambience_volume
        self.ring_volume = ring_volume

        self.ring_min_interval = ring_min_interval
        self.ring_max_interval = ring_max_interval

        self.ambience = self._load_audio(
            self.ambience_path,
            "ambience",
        )

        self.ring = self._load_audio(
            self.ring_path,
            "ring",
        )

        # Current position inside the ambience file.
        self.ambience_position = 0

        # Current position inside the ring file.
        self.ring_position = 0
        self.ring_active = False

        # Audio-stream position.
        #
        # Unlike the previous implementation, ring scheduling
        # is based on audio samples instead of time.monotonic().
        self.stream_position = 0

        self.next_ring_position = (
            random.uniform(
                self.ring_min_interval,
                self.ring_max_interval,
            )
            * self.SAMPLE_RATE
        )

    def _load_audio(
        self,
        path: Path,
        name: str,
    ) -> np.ndarray:
        """
        Load a WAV file.

        Expected:
            24 kHz
            mono
            16-bit PCM
        """

        if not path.exists():
            raise FileNotFoundError(
                f"{name} audio file not found: {path}"
            )

        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frame_count = wav.getnframes()

            frames = wav.readframes(frame_count)

        if channels != 1:
            raise ValueError(
                f"{name} must be mono. "
                f"Got {channels} channels: {path}"
            )

        if sample_width != self.SAMPLE_WIDTH:
            raise ValueError(
                f"{name} must be 16-bit PCM. "
                f"Got {sample_width * 8}-bit: {path}"
            )

        if sample_rate != self.SAMPLE_RATE:
            raise ValueError(
                f"{name} must be {self.SAMPLE_RATE} Hz. "
                f"Got {sample_rate} Hz: {path}"
            )

        if not frames:
            raise ValueError(
                f"{name} file contains no audio: {path}"
            )

        return np.frombuffer(
            frames,
            dtype="<i2",
        ).astype(np.float32)

    def next_frame(
        self,
        num_samples: int,
    ) -> np.ndarray:
        """
        Generate the next continuous background frame.

        Returns:
            float32 mono audio samples.

        The returned audio contains:

            ambience + phone ring (when active)

        It does NOT contain TTS.
        """

        if num_samples <= 0:
            return np.empty(
                0,
                dtype=np.float32,
            )

        ambience = self._next_ambience(
            num_samples
        )

        ring = self._next_ring(
            num_samples
        )

        background = (
            ambience * self.ambience_volume
            + ring * self.ring_volume
        )

        self.stream_position += num_samples

        return background

    def _next_ambience(
        self,
        num_samples: int,
    ) -> np.ndarray:
        """
        Get the next section of the ambience.

        The ambience loops forever.
        """

        result = np.empty(
            num_samples,
            dtype=np.float32,
        )

        remaining = num_samples
        write_position = 0

        while remaining > 0:

            available = (
                len(self.ambience)
                - self.ambience_position
            )

            count = min(
                remaining,
                available,
            )

            result[
                write_position:
                write_position + count
            ] = self.ambience[
                self.ambience_position:
                self.ambience_position + count
            ]

            self.ambience_position += count
            write_position += count
            remaining -= count

            if self.ambience_position >= len(
                self.ambience
            ):
                self.ambience_position = 0

        return result

    def _next_ring(
        self,
        num_samples: int,
    ) -> np.ndarray:
        """
        Generate the phone-ring portion of the frame.

        Ring timing is based on the continuous audio
        stream position, so it works even while:

            - the user is speaking
            - STT is processing
            - the LLM is extracting
            - TTS is being generated
            - no TTS is playing
        """

        result = np.zeros(
            num_samples,
            dtype=np.float32,
        )

        frame_start = self.stream_position
        frame_end = (
            self.stream_position
            + num_samples
        )

        # Start a new ring if its scheduled position
        # falls inside this frame.
        if (
            not self.ring_active
            and frame_end >= self.next_ring_position
        ):
            self.ring_active = True

            ring_offset = max(
                0,
                int(
                    self.next_ring_position
                    - frame_start
                ),
            )

            self.ring_position = 0
        else:
            ring_offset = 0

        if not self.ring_active:
            return result

        available_ring = (
            len(self.ring)
            - self.ring_position
        )

        available_output = (
            num_samples
            - ring_offset
        )

        count = min(
            available_ring,
            available_output,
        )

        if count > 0:
            result[
                ring_offset:
                ring_offset + count
            ] = self.ring[
                self.ring_position:
                self.ring_position + count
            ]

            self.ring_position += count

        if self.ring_position >= len(self.ring):
            self.ring_active = False
            self.ring_position = 0

            self.next_ring_position = (
                self.stream_position
                + num_samples
                + random.uniform(
                    self.ring_min_interval,
                    self.ring_max_interval,
                )
                * self.SAMPLE_RATE
            )

        return result