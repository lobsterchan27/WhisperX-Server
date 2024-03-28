import os
import uvicorn
import configparser

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import HttpUrl
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from audio_processor import AudioProcessor
from schema import RequestParam
from video_download import save_upload_file, save_link
from settings import HF_TOKEN

import tts_functions





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

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ThreadPoolExecutor(max_workers=5)
    app.state.audio_processor = AudioProcessor(model_settings,
                                 align=config.getboolean('Model Settings', 'align'),
                                 diarization=config.getboolean('Model Settings', 'diarize'),
                                 HF_TOKEN=HF_TOKEN)
    # tts_functions.create_tts()
    yield
    app.state.audio_processor = None
    app.state.executor.shutdown(wait=True)
    app.state.audio_processor.clean_up()

app = FastAPI(lifespan=lifespan)
#load tts model + rvc model


@app.post("/api/transcribe/file")
async def transcribe_file(file: UploadFile = File(...),
                          language: Optional[str] = Form(None),
                          text2speech: Optional[bool] = Form(False),
                          segment_audio: Optional[bool] = Form(False),
                          translate: Optional[bool] = Form(False),
                          get_video: Optional[bool] = Form(False)):
    param = RequestParam(language=language,
                         text2speech=text2speech,
                         segment_audio=segment_audio,
                         translate=translate,
                         get_video=get_video)
    file_path = await save_upload_file(file)
    result, _ = app.state.audio_processor.process(file_path, param)
    return result


@app.post("/api/transcribe/url")
async def transcribe_url(url: HttpUrl,
                         language: Optional[str] = None,
                         text2speech: Optional[bool] = False,
                         segment_audio: Optional[bool] = False,
                         translate: Optional[bool] = False,
                         get_video: Optional[bool] = False):
    param = RequestParam(language=language,
                         text2speech=text2speech,
                         segment_audio=segment_audio,
                         translate=translate,
                         get_video=get_video)
    file_path = await save_link(app, url)
    result = app.state.audio_processor.process(file_path, param)
    return result


@app.post("/api/text2speech")
def text2speech(text: str):
    return "Not implemented yet"


if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))