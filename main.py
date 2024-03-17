import os
import whisperx
import uvicorn
import configparser
import tempfile
from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from model_loader import AudioProcessor, AudioParams, Response
from starlette.concurrency import run_in_threadpool
from whisperx.types import TranscriptionResult

app = FastAPI()

# Load the environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

config = configparser.ConfigParser()
config.read('config.ini')
model_settings = {
    "device": config.get('Model Settings', 'device', fallback='cuda'),
    "compute_type": config.get('Model Settings', 'compute_type', fallback='float32'),
    "language": config.get('Model Settings', 'language', fallback=None),
    "whisper_arch": config.get('Model Settings', 'whisper_arch', fallback='large-v3'),
    "asr_options": {
        "word_timestamps": config.getboolean('Model Settings', 'word_timestamps', fallback=True)
    },
}

audio_processor = AudioProcessor(model_settings,
                                 align=config.getboolean('Model Settings', 'align'),
                                 diarization=config.getboolean('Model Settings', 'diarize'),
                                 HF_TOKEN=HF_TOKEN)


async def save_upload_file(upload_file: UploadFile) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await upload_file.read())
            return temp_file.name
    except Exception as e:
        print(f"Failed to save upload file: {e}")
        return None

    
@app.post("/api/audio")
async def process(file: Union[str, UploadFile], param: AudioParams=None):
    # If the file is an uploaded file, save it and get the file path
    if isinstance(file, UploadFile):
        file = await save_upload_file(file)
    
    # If a language parameter is provided, update the language of the audio processor
    if param.language:
        audio_processor.update_language(param.language)
    
    # Load the audio file
    batch_size = 16
    audio = whisperx.load_audio(file)
    
    # Transcribe the audio file
    result = audio_processor.model.transcribe(audio, batch_size=batch_size)
    
    # If the segment_audio parameter is true, segment the audio
    if param.segment_audio:
        result = whisperx.align(result["segments"], audio_processor.align_model, audio_processor.metadata, audio, model_settings["device"], return_char_alignments=False)
    
    # If the diarize parameter is true, diarize the audio
    if param.diarize:
        diarize_segments = audio_processor.diarize_model(audio)
        result = whisperx.assign_word_speakers(result, diarize_segments)
    
    return result


@app.post("/api/text2speech")
def process(text: str):
    return "Not implemented yet"


if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))