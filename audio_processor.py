import whisperx
import gc
import torch
import time

from schema import RequestParam

class AudioProcessor:
    def __init__(self, model_settings, align=False, diarization=False, HF_TOKEN=None):
        self.model = None
        self.align_model = None
        self.diarize_model = None
        self.metadata = None
        self.device = 'cuda'
        
        start_time = time.time()
        self.load_whisperx(model_settings)
        print("Whisperx model load time: ", time.time() - start_time)
        if align:
            start_time = time.time()
            self.load_align(model_settings["language"], model_settings["device"])
            print("Align model load time: ", time.time() - start_time)
        if diarization:
            start_time = time.time()
            self.load_diarization(HF_TOKEN, model_settings["device"])
            print("Diarization model load time: ", time.time() - start_time)

    async def process(self, file: str, audio_params: RequestParam):
        batch_size = 16
        audio = whisperx.load_audio(file)

        # Check task and language
        self.set_task(audio_params.translate)
        self.update_language(audio_params.language)

        # Transcribe audio
        print("Transcribing audio...")
        start_time = time.time()
        result = self.model.transcribe(audio, batch_size=batch_size)
        print("Transcription time: ", time.time() - start_time)

        # Word alignment
        start_time = time.time()
        result = whisperx.align(result["segments"], self.align_model, self.metadata, audio, device=self.device)
        print("Alignment time: ", time.time() - start_time)

        # Diarization
        start_time = time.time()
        diarization = self.diarize_model(audio)
        print("Diarization time: ", time.time() - start_time)

        # Assign speakers
        result = whisperx.assign_word_speakers(diarization, result)
        self.clean_up()
        return result
    
    def set_task(self, translate: bool):
        if translate:
            self.model.task = 'translate'
        else:
            self.model.task = 'transcribe'
    
    def update_language(self, language: str):
        if language:
            self.model.language = language

    def load_whisperx(self, model_settings: dict):
        self.model = whisperx.load_model(device=model_settings["device"],
                                    compute_type=model_settings["compute_type"],
                                    language=model_settings["language"],
                                    whisper_arch=model_settings["whisper_arch"])

    def load_align(self, language_code: str, device: str):
        self.align_model, self.metadata = whisperx.load_align_model(language_code=language_code, device=device)

    def load_diarization(self, use_auth_token: str, device: str):
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)

    def clean_up(self, final=False):
        torch.cuda.empty_cache()
        if final:
            gc.collect()
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'align_model'):
                del self.align_model
            if hasattr(self, 'diarize_model'):
                del self.diarize_model