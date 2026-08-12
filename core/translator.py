import os

from sarvamai import SarvamAI

api_key = os.environ.get("SARVAM_API_KEY")

_client = None

DEFAULT_SOURCE_LANGUAGE_CODE = "ta-IN"
DEFAULT_TARGET_LANGUAGE_CODE = "en-IN"
DEFAULT_MODEL = "mayura:v1"
DEFAULT_NUMERALS_FORMAT = "native"
DEFAULT_MODE = "formal"


def get_client() -> SarvamAI:
    global _client
    if _client is None:
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set in the environment variables.")
        _client = SarvamAI(api_key=api_key)
    return _client


def extract_translated_text(response) -> str:
    if isinstance(response, str):
        return response

    for attribute_name in ("translated_text", "text", "translation"):
        if hasattr(response, attribute_name):
            value = getattr(response, attribute_name)
            if value:
                return str(value)

    if isinstance(response, dict):
        for key in ("translated_text", "text", "translation"):
            value = response.get(key)
            if value:
                return str(value)

    return str(response)


def translate_text_chunk(
    text: str,
    source_language_code: str = DEFAULT_SOURCE_LANGUAGE_CODE,
    target_language_code: str = DEFAULT_TARGET_LANGUAGE_CODE,
    model: str = DEFAULT_MODEL,
    numerals_format: str = DEFAULT_NUMERALS_FORMAT,
    mode: str = DEFAULT_MODE,
) -> str:
    client = get_client()
    response = client.text.translate(
        text=text,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
        model=model,
        numerals_format=numerals_format,
        mode=mode,
    )
    return extract_translated_text(response)


def translate_all_chunks(
    chunks: list,
    source_language_code: str = DEFAULT_SOURCE_LANGUAGE_CODE,
    target_language_code: str = DEFAULT_TARGET_LANGUAGE_CODE,
) -> str:
    full_translation = ""
    for i, chunk in enumerate(chunks):
        print(f"Translating chunk {i + 1}/{len(chunks)}")
        translation = translate_text_chunk(
            chunk,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )
        full_translation += translation + " "
    print(f"Full translation: {full_translation}")
    return full_translation.strip()


def process_translation(
    transcribed_text: str,
    source_language_code: str = DEFAULT_SOURCE_LANGUAGE_CODE,
    target_language_code: str = DEFAULT_TARGET_LANGUAGE_CODE,
    chunk_size: int = 100,
) -> str:
    print("Chunking the transcribed text into smaller segments for translation.")
    chunks = [
        transcribed_text[i : i + chunk_size]
        for i in range(0, len(transcribed_text), chunk_size)
    ]
    return translate_all_chunks(
        chunks,
        source_language_code=source_language_code,
        target_language_code=target_language_code,
    )
