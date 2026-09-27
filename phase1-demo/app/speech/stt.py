import os
import tempfile
import cohere
from fastapi import UploadFile
from pydub import AudioSegment

COHERE_MODEL = "cohere-transcribe-arabic-07-2026"

_co = None

def get_client() -> cohere.ClientV2:
    global _co
    if _co is None:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise RuntimeError("COHERE_API_KEY environment variable is not set.")
        _co = cohere.ClientV2(api_key=api_key)
    return _co

async def transcribe_audio(audio: UploadFile, language: str = "ar") -> str:
    # Write uploaded file to a temporary file in chunks to avoid loading large files into memory
    with tempfile.NamedTemporaryFile(delete=False, suffix='_input') as tmp_in:
        try:
            while chunk := await audio.read(1024 * 1024):  # Read 1MB chunks
                tmp_in.write(chunk)
        finally:
            await audio.close()
        tmp_in_path = tmp_in.name

    # Convert to WAV format (required by Cohere)
    tmp_out_path = tmp_in_path + ".wav"
    try:
        # Load audio from the temporary input file and export as WAV
        sound = AudioSegment.from_file(tmp_in_path)
        sound.export(tmp_out_path, format="wav")

        # Transcribe with Cohere
        co = get_client()
        with open(tmp_out_path, "rb") as f:
            response = co.audio.transcriptions.create(
                model=COHERE_MODEL,
                language=language,
                file=f,
            )
        return response.text
    finally:
        # Clean up temporary files
        try:
            os.remove(tmp_in_path)
        except OSError:
            pass
        try:
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
        except OSError:
            pass