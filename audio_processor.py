from typing import Optional
from pydantic import BaseModel
import whisperx

class AudioParams(BaseModel):
    language: str
    segment_audio: Optional[bool] = False
    translate: Optional[bool] = False

class AudioProcessor:
    def __init__(self, model_settings, align=False, diarization=False, HF_TOKEN=None):
        self.model = load_whisperx(model_settings)
        if align:
            self.align_model, self.metadata = load_align(model_settings["language"], model_settings["device"])
        if diarization:
            self.diarize_model = load_diarization(HF_TOKEN, model_settings["device"])

    def process(self, file: str, audio_params: AudioParams):
        batch_size = 16
        audio = whisperx.load_audio(file)

        # Check task and language
        self.set_task(audio_params.translate)
        self.update_language(audio_params.language)

        result = self.model.transcribe(audio, batch_size=batch_size)
        return result, audio
    
    def set_task(self, translate: bool):
        if translate:
            self.model.task = 'translate'
        else:
            self.model.task = 'transcribe'
    
    def update_language(self, language: str):
        if language:
            self.model.language = language

def load_whisperx(model_settings: dict):
    model = whisperx.load_model(device=model_settings["device"],
                                compute_type=model_settings["compute_type"],
                                language=model_settings["language"],
                                whisper_arch=model_settings["whisper_arch"],
                                asr_options=model_settings["asr_options"])
    return model

def load_align(language_code: str, device: str):
    align_model, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    return align_model, metadata

def load_diarization(use_auth_token: str, device: str):
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)
    return diarize_model