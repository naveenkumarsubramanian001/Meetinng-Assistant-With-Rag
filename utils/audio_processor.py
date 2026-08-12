import yt_dlp
from pydub import AudioSegment
import os


DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_audio_from_youtube(url : str) ->str :
    output_path = os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s')
    ydl_opts={
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors":[
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url,download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav").replace(".mp3", ".wav")
    return filename


def convert_to_wav(file_path: str) -> str:
    output_path = os.path.splitext(file_path)[0] + "_converted.wav"
    if os.path.exists(file_path):
        audio = AudioSegment.from_file(file_path)
        os.remove(file_path)  # Remove the original file after conversion
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="wav")
        return output_path
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")
    


def chunk_audio(wav_path : str, chunk_minutes : int= 3) -> list:
    if  os.path.exists(wav_path): 
        audio = AudioSegment.from_wav(wav_path)
        chunk_length_ms = chunk_minutes * 60 * 1000  # Convert minutes to milliseconds
        chunks= []
        for i,start in enumerate(range(0, len(audio), chunk_length_ms)):
            chunk = audio[start:start + chunk_length_ms]
            chunk_path = os.path.splitext(wav_path)[0] + f"_chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
        return chunks 
    else:
        raise FileNotFoundError(f"File {wav_path} does not exist.")


def process_input(source: str) -> list:
    if source.startswith("http"):
        print(f"Downloading audio from YouTube URL: {source}")
        audio_path = download_audio_from_youtube(source)
        audio_path = convert_to_wav(audio_path)
    else:
        print(f"Processing local audio file: {source}")
        audio_path = convert_to_wav(source)
    
    print(f"Chunking audio file: {audio_path}")
    chunks = chunk_audio(audio_path)
    print(f"Created {len(chunks)} chunks.")
    return chunks

