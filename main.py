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
    "asr_options": {"word_timestamps": config.getboolean('Model Settings', 'word_timestamps', fallback=True)},
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
    if isinstance(file, UploadFile):
        file = await run_in_threadpool(save_upload_file, file)  # run_in_threadpool is used to run sync function in async context
    if param.language:
        audio_processor.update_language(param.language)
    batch_size = 16
    audio = whisperx.load_audio(file)
    result = Response(**audio_processor.model.transcribe(audio, batch_size=batch_size))
    return result

@app.post("/api/text2speech")
def process(text: str):
    return "Not implemented yet"

if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))