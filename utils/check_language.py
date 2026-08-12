import random
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


def create_random_line_chunks(transcribed_text: str, num_chunks: int = 10) -> str:
    lines = transcribed_text.splitlines()
    lines = [line for line in lines if line.strip()]

    if not lines:
        return ""

    random_lines_from_transcription = random.sample(lines, min(num_chunks, len(lines)))
    line_chunk = "\n".join(random_lines_from_transcription)

    return line_chunk


def find_language(line_chunk: str) -> str:
    try:
        lang = detect(line_chunk)
    except Exception:
        return "unknown"

    if  lang.lower() in ["en", "ta"]:
        return lang.lower()
    else:
        return "unknown"


