# Video AI

Video AI is a Python pipeline that downloads or reads audio, transcribes it with Whisper, translates Tamil to English when needed, extracts meeting insights, generates a summary and title, and supports retrieval-augmented question answering over the transcript.

## Features

- Audio ingestion from a YouTube URL or local file
- Chunked transcription with `faster-whisper`
- Tamil-to-English translation with Sarvam AI
- Extraction of action items, decisions, and questions with LangChain prompts
- Meeting summary and short title generation
- Vector-store-backed RAG chat over the transcript

## Requirements

- Python 3.11 or newer
- `ffmpeg` installed and available on your `PATH`
- API keys for the external services you want to use:
	- `MISTRALAI_API_KEY` for summarization, extraction, title generation, and RAG
	- `SARVAM_API_KEY` for translation

## Installation

1. Create and activate a virtual environment.
2. Install the project dependencies.

```bash
python -m pip install -U pip
python -m pip install -e .
```

If you use `uv`, you can install dependencies with:

```bash
uv sync
```

## Environment Variables

Create a `.env` file in the project root with the keys you need:

```env
MISTRALAI_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
WHISPER_MODEL=small
WHISPER_LANGUAGE=ta
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

## Usage

Run the main pipeline with:

```bash
python main.py
```

By default, `main.py` downloads/transcribes the sample YouTube source in the file, prints the title, summary, transcription, action items, decisions, and questions, then opens an interactive RAG chat loop.

You can also reuse the pipeline from your own code:

```python
from main import run_pipeline

results = run_pipeline("https://youtu.be/your-video-id")
print(results["summary"])
```

## Project Structure

- `main.py` - end-to-end pipeline and RAG chat loop
- `core/transcriber.py` - Whisper transcription
- `core/translator.py` - Sarvam translation helpers
- `core/summariser.py` - summary and title generation
- `core/extractor.py` - action item, decision, and question extraction
- `core/rag_engine.py` - vector store creation and question answering
- `core/vector_store.py` - Chroma vector store helpers
- `utils/audio_processor.py` - download, convert, and chunk audio
- `utils/check_language.py` - language detection helpers

## Notes

- The first run may take longer while models and dependencies are downloaded.
- If you update `.env`, restart the Python process so the new environment variables are picked up.
- The project writes audio chunks to `downloads/` and persists the vector store in `vector_db/`.

