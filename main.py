import os
import gc
import uuid
import json
import asyncio
import uvicorn
import configparser
import time

from settings import HOSTNAME, PORT, AudioProcessorSettings
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from pydantic import HttpUrl
from contextlib import asynccontextmanager
from util import chunk_segments, prepare_for_align, clean_up
from fastapi.responses import Response

from audio_processor import AudioProcessor
from schema import RequestParam, MultipartResponse, TTSRequest
from video_download import save_upload_file, save_link, generate_storyboards

from tts_functions import generate_tts, to_wav, create_tts
from rvc_processing import VCWrapper

config = configparser.ConfigParser()
config.read('config.ini')

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    app.state.audio_processor = AudioProcessor(model_settings=AudioProcessorSettings())
    # app.state.audio_processor.load_whisperx()
    app.state.audio_processor.load_align()
    # app.state.audio_processor.load_diarization()
    start_time = time.time()
    app.state.tts = create_tts()
    print(f"Tortoise Start Time: {time.time() - start_time}")
    app.state.vc = VCWrapper()
    app.state.lock = asyncio.Lock()
    yield
    print("Shutting down")
    app.state.audio_processor.clean_up(final=True)
    app.state.audio_processor = None
    app.state.vc = None
    app.state.tts = None

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def process_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"Processing time: {process_time}")
    return response


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
                         segment_length: Optional[int] = Form(30),
                         scene_threshold: Optional[float] = Form(0.02),
                         minimum_interval: Optional[float] = Form(0),
                         fixed_interval: Optional[float] = Form(None),
                         translate: Optional[bool] = Form(False),
                         get_video: Optional[bool] = Form(True)):
    param = RequestParam(language=language,
                         text2speech=text2speech,
                         segment_length=segment_length,
                         scene_threshold=scene_threshold,
                         minimum_interval=minimum_interval,
                         fixed_interval=fixed_interval,
                         translate=translate,
                         get_video=get_video)
    file_path = await save_link(url, param)

    async with app.state.lock:
        clean_up()
        transcript = await app.state.audio_processor.process(file_path.audio, param)
        clean_up()

        storyboards = None
        if param.get_video:
            storyboards = await generate_storyboards(file_path.video, param)

    segments = transcript['segments']
    transform_func = lambda segment: {key: segment[key] for key in segment if key != 'words'}

    async def generate_data():
        if storyboards:
            for index, (storyboard, chunk) in enumerate(zip(storyboards, chunk_segments(segments, param.segment_length, lambda x: x['start'], transform_func))):
                filename = os.path.basename(storyboard)
                yield ("image/webp", storyboard)
                yield ("application/json", {str(index): {"filename": filename, "segments": chunk}})

    headers = {"Base-Filename": file_path.basename}
    return MultipartResponse()(generate_data(), headers)

# #incomplete
# @app.get("/api/transcribe/stream")
# async def transcribe_url(
#     url: HttpUrl,
#     language: Optional[str] = Form(None),
#     text2speech: Optional[bool] = Form(False),
#     segment_length: Optional[bool] = Form(False),
#     translate: Optional[bool] = Form(False),
#     get_video: Optional[bool] = Form(False)):
#     params = RequestParam(language=language,
#                           text2speech=text2speech,
#                           segment_length=segment_length,
#                           translate=translate,
#                           get_video=get_video)
#     async def generate_chunks(url, **params):
#         save_path = await save_link(url, params)
#         storyboards = []
#         storyboards = await generate_storyboards(save_path)
#         result, _ = await asyncio.to_thread(app.state.audio_processor.process(file_path, params))
#         for i in range(10):
#             chunk = f"Chunk {i}"
#             yield chunk
#             await asyncio.sleep(1)  # Simulating asynchronous processing delay

#     chunks_generator = generate_chunks(file, **params)

@app.post("/api/text2speech/align")
async def text2speech(request: TTSRequest):

    async with app.state.lock:
        clean_up()
        result, duration = generate_tts(app.state.tts, request.prompt, request.voice)
        clean_up()
        result, samplerate = app.state.vc.vc_process(result)
        clean_up()

        segments = [{
            'start': 0.0,
            'end': duration,
            'text': request.prompt
        }]
        segments = app.state.audio_processor.alignment(segments, prepare_for_align(result))
        clean_up()
    result = to_wav(result, samplerate)

    async def generate_data():
        yield ("application/json", segments['segments'])
        yield ("audio/wav", result)
        
    headers = {'Voice': request.voice}
    return MultipartResponse()(generate_data(), headers)

@app.post("/api/text2speech")
async def text2speech(request: TTSRequest):
    async with app.state.lock:
        clean_up()
        result, duration = generate_tts(app.state.tts, request.prompt, request.voice)
        clean_up()
        result, samplerate = app.state.vc.vc_process(result)
        clean_up()
    result = to_wav(result, samplerate)
    headers = {'Voice': request.voice}
    return Response(content=result, media_type="audio/wav", headers=headers)

@app.post("/api/rvc")
async def rvc():
    return "Not implemented yet"



if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))