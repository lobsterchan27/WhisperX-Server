import os
import uuid
import json
import asyncio
import uvicorn
import configparser
import time

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Request
from pydantic import HttpUrl
from contextlib import asynccontextmanager
from util import chunk_segments
from fastapi.responses import Response

from audio_processor import AudioProcessor
from schema import RequestParam, MultipartResponse, TTSRequest
from video_download import save_upload_file, save_link, generate_storyboards
from settings import HF_TOKEN

from tts_functions import generate_tts, to_wav, create_tts
from rvc_processing import VCWrapper

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
    start_time = time.time()
    app.state.tts = create_tts()
    print(f"Tortoise Start Time: {time.time() - start_time}")
    app.state.vc = VCWrapper()
    yield
    print("Shutting down")
    app.state.audio_processor.clean_up(final=True)
    app.state.audio_processor = None
    app.state.vc = None
    app.state.tts = None

app = FastAPI(lifespan=lifespan)
#load tts model + rvc model
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

    tasks = [app.state.audio_processor.process(file_path.audio, param)]
    if param.get_video:
        tasks.append(generate_storyboards(file_path.video, param))
    results = await asyncio.gather(*tasks)
    transcript = results[0]
    storyboards = results[1] if param.get_video else None

    segments = transcript['segments']
    transform_func = lambda segment: {key: segment[key] for key in segment if key != 'words'}

    async def generate_data():
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

# text2speech tortoise > rvc
@app.post("/api/text2speech")
async def text2speech(request: TTSRequest):
    result = generate_tts(app.state.tts, request.prompt, request.voice)
    result, samplerate = app.state.vc.vc_process(result)
    result = to_wav(result, samplerate)
    headers = {'Voice': request.voice}
    return Response(content=result, media_type="audio/wav", headers=headers)

@app.post("/api/text2speech/whisperx")
async def text2speech_whisperx(request: TTSRequest):
    # Generate the TTS audio
    result = generate_tts(app.state.tts, request.prompt, request.voice)
    result, samplerate = app.state.vc.vc_process(result)
    result = to_wav(result, samplerate)

    # Process the audio with WhisperX
    transcription = {'message': "WhisperX is not implemented yet"}

    # Create the content generator
    async def content():
        yield "application/json", transcription
        yield "audio/wav", result
    
    headers = {'Voice': request.voice}
    return MultipartResponse()(content(), headers=headers)

@app.post("/api/rvc")
async def rvc():
    return "Not implemented yet"



if __name__ == "__main__":
    uvicorn.run(app,
                host=config.get('Network', 'host'),
                port=config.getint('Network', 'port'))