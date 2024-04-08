import os
import uuid
import json
import asyncio
import uvicorn
import configparser
import time

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import HttpUrl
from contextlib import asynccontextmanager
from util import chunk_segments

from audio_processor import AudioProcessor
from schema import RequestParam, MultipartResponse
from video_download import save_upload_file, save_link, generate_storyboards
from settings import HF_TOKEN

import tts_functions

config = configparser.ConfigParser()
config.read('config.ini')
model_settings = {
    "device": config.get('Model Settings', 'device', fallback='cuda'),
    "compute_type": config.get('Model Settings', 'compute_type', fallback='float32'),
    "language": config.get('Model Settings', 'language', fallback=None),
    "whisper_arch": config.get('Model Settings', 'whisper_arch', fallback='large-v3')
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    app.state.audio_processor = AudioProcessor(model_settings,
                                 align=config.getboolean('Model Settings', 'align'),
                                 diarization=config.getboolean('Model Settings', 'diarization'),
                                 HF_TOKEN=HF_TOKEN)
    # tts_functions.create_tts()
    yield
    print("Shutting down")
    app.state.audio_processor.clean_up(True)
    app.state.audio_processor = None

app = FastAPI(lifespan=lifespan)
#load tts model + rvc model


@app.post("/api/transcribe/file")
async def transcribe_file(file: UploadFile = File(...),
                          language: Optional[str] = Form(None),
                          text2speech: Optional[bool] = Form(False),
                          segment_length: Optional[int] = Form(False),
                          translate: Optional[bool] = Form(False),
                          get_video: Optional[bool] = Form(False)):
    param = RequestParam(language=language,
                         text2speech=text2speech,
                         segment_length=segment_length,
                         translate=translate,
                         get_video=get_video)
    file_path = await save_upload_file(file)
    transcription = app.state.audio_processor.process(file_path, param)
    return transcription

@app.post("/api/transcribe/url")
async def transcribe_url(url: HttpUrl = Form(...),
                         language: Optional[str] = Form(None),
                         text2speech: Optional[bool] = Form(False),
                         segment_length: Optional[int] = Form(False),
                         translate: Optional[bool] = Form(False),
                         get_video: Optional[bool] = Form(False)):
    param = RequestParam(language=language,
                         text2speech=text2speech,
                         segment_length=segment_length,
                         translate=translate,
                         get_video=get_video)
    file_path = await save_link(url, param)
    if param.get_video:
        storyboards = await generate_storyboards(file_path.video, param)
    transcript = await app.state.audio_processor.process(file_path.audio, param)

    segments = transcript['segments']
    transform_func = lambda segment: {key: segment[key] for key in segment if key != 'words'}

    async def generate_data():
        for storyboard, chunk in zip(storyboards, chunk_segments(segments, param.segment_length, lambda x: x['start'], transform_func)):
            yield ("text/plain", storyboard)  # assuming storyboard is image data
            yield ("application/json", {"Segments": chunk})
    
    return MultipartResponse()(generate_data())

#incomplete
@app.get("/api/transcribe/stream")
async def transcribe_url(
    url: HttpUrl,
    language: Optional[str] = Form(None),
    text2speech: Optional[bool] = Form(False),
    segment_length: Optional[bool] = Form(False),
    translate: Optional[bool] = Form(False),
    get_video: Optional[bool] = Form(False)):
    params = RequestParam(language=language,
                          text2speech=text2speech,
                          segment_length=segment_length,
                          translate=translate,
                          get_video=get_video)
    async def generate_chunks(url, **params):
        save_path = await save_link(url, params)
        storyboards = []
        storyboards = await generate_storyboards(save_path)
        result, _ = await asyncio.to_thread(app.state.audio_processor.process(file_path, params))
        for i in range(10):
            chunk = f"Chunk {i}"
            yield chunk
            await asyncio.sleep(1)  # Simulating asynchronous processing delay

    chunks_generator = generate_chunks(file, **params)

@app.post("/api/text2speech")
async def text2speech(text: str):
    return "Not implemented yet"

@app.post("/api/rvc")
async def rvc():
    return "Not implemented yet"


if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))