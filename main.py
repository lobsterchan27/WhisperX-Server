import os
import whisperx
import uvicorn
from dotenv import load_dotenv
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

# Load the environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI()

class AudioParams(BaseModel):
    language: str
    segment_audio: Optional[bool] = False

class Segment(BaseModel):
    text: str
    start: float
    end: float

class Response(BaseModel):
    segments: List[Segment]
    language: str

# Model settings
model_settings = {
    "device": "cuda",
    "compute_type": "float32",
    "language": "en",
    "whisper_arch": "large-v3",
    "asr_options": {"word_timestamps": True}
}

# Load the model
model = whisperx.load_model(**model_settings)
print("Model loaded")
# align_model, metadata = whisperx.load_align_model(language_code=model_settings["language"], device=model_settings["device"])
# diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=model_settings["device"])

@app.post("/api/audio")
def process(file: str, task_list: AudioParams=None):
# def process(file: UploadFile=File(...), task_list: TaskList=None):
    batch_size = 16
    audio = whisperx.load_audio(file)
    result = Response(**model.transcribe(audio, batch_size=batch_size))
    if task_list and task_list.segment_audio:
        result = whisperx.align(result["segments"], align_model, audio, device, return_char_alignments=False)
    return result

def transcribe(model, video_path):
    batch_size = 16
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=batch_size)
    return result

if __name__ == "__main__":

    host_config = {
    "host": "127.0.0.1",
    "port": 8127
    }

    uvicorn.run(app, **host_config)