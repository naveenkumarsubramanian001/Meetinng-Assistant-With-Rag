import os
from faster_whisper import WhisperModel
from utils.audio_processor import process_input

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")  
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ta")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")


_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print(f"Whisper model {WHISPER_MODEL} loaded successfully.")
    return _model 

def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = get_model()
    task = "translate" if translate else "transcribe"

    segments, _info = model.transcribe(
        chunk_path,
        task=task,
        language=WHISPER_LANGUAGE if translate else None,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}: {chunk}")
        transcription = transcribe_chunk(chunk, translate = translate)
        full_transcription += transcription + " "

    print(f"Full transcription: {full_transcription}")
    return full_transcription.strip()
    