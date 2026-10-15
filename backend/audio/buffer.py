# separates and creates small audio chunks

class AudioBuffer:
    def __init__(self):
        self.chunks = []

    def add(self, audio_chunk: bytes):
        self.chunks.append(audio_chunk)

    def get_audio(self) -> bytes:
        return b"".join(self.chunks)

    def size(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    def clear(self):
        self.chunks.clear()

    def is_empty(self) -> bool:
        return len(self.chunks) == 0