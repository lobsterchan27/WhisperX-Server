import os
import uvicorn
import configparser

from typing import Union
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile

from audio_processor import AudioProcessor
from schema import RequestParam
from video_download import save_upload_file, save_link

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

#load tts model + rvc model


@app.post("/api/transcribe")
async def transcribe(file: Union[str, UploadFile], param: RequestParam=None):
    # If the file is an uploaded file, save it and get the file path
    if isinstance(file, UploadFile):
        file = await save_upload_file(file)
    elif isinstance(file, str):
        file = await save_link(file)
    
    # Add asynchoronous tasks LLM caching, process audio, and return result. Refactor functions to be async
    # Process the audio
    result = audio_processor.process(file, param)
    
    return result


@app.post("/api/text2speech")
def text2speech(text: str):
    return "Not implemented yet"


if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))