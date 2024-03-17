from typing import Optional, List, Union
from pydantic import BaseModel
import whisperx

class AudioParams(BaseModel):
    language: str
    segment_audio: Optional[bool] = False
    
class AudioProcessor:
    def __init__(self, model_settings, align=False, diarization=False, HF_TOKEN=None):
        self.model = load_whisperx(**model_settings)
        if align:
            self.align_model, self.metadata = load_align(model_settings["language"], model_settings["device"])
        if diarization:
            self.diarize_model = load_diarization(HF_TOKEN, model_settings["device"])

    def process(self, file: str, param: AudioParams=None):
        batch_size = 16
        audio = whisperx.load_audio(file)
        result = Response(**self.model.transcribe(audio, batch_size=batch_size))
        return result
    
    def update_language(self, new_language):
        self.model.language = new_language

def load_whisperx(device: str = "cuda",
                  compute_type: str = "float32",
                  language: Optional[str] = None,
                  whisper_arch: str = "large-v3",
                  asr_options: dict = {"word_timestamps": True}):
    
    model = whisperx.load_model(device=device,
                                compute_type=compute_type,
                                language=language,
                                whisper_arch=whisper_arch,
                                asr_options=asr_options)
    return model

def load_align(language_code: str, device: str):
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    return align_model, metadata

def load_diarization(use_auth_token: str, device: str):
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)
    return diarize_model