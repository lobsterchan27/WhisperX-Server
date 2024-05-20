import whisperx
import gc
import torch
import time
import asyncio

from schema import RequestParam
from settings import AudioProcessorSettings

from settings import HF_TOKEN, COMPUTE_TYPE

class AudioProcessor:
    def __init__(self, model_settings = AudioProcessorSettings):
        self.model_settings = model_settings
        self.model = None
        self.align_model = None
        self.diarize_model = None
        self.diarize_token = HF_TOKEN
        self.metadata = None

        self.device = model_settings.device
        self.compute_type = model_settings.compute_type
        self.language = model_settings.language
        self.whisper_arch = model_settings.whisper_arch
        self.batch_size = model_settings.batch_size

    def load_whisperx(self):
        if self.model is not None:
            return

        start_time = time.time()
        self.model = whisperx.load_model(device=self.device,
                                        compute_type=self.compute_type,
                                        language=self.language,
                                        whisper_arch=self.whisper_arch)
        self.clean_up()
        print("Whisperx model load time: ", time.time() - start_time)

    def load_align(self):
        if self.align_model is not None:
            return
        
        start_time = time.time()
        self.align_model, self.metadata = whisperx.load_align_model(language_code=self.language, device=self.device)
        print("Align model load time: ", time.time() - start_time)

    def load_diarization(self):
        if self.diarize_model is not None:
            return
        
        start_time = time.time()
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.diarize_token, device=self.device)
        print("Diarization model load time: ", time.time() - start_time)

    def unload_whisperx(self):
        if self.model is not None:
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

    def unload_align(self):
        if self.align_model is not None:
            self.align_model = None
            torch.cuda.empty_cache()
            gc.collect()

    def unload_diarization(self):
        if self.diarize_model is not None:
            self.diarize_model = None
            torch.cuda.empty_cache()
            gc.collect()

    async def process(self, file: str, param: RequestParam):
        audio = whisperx.load_audio(file)

        # Check task and language
        self.set_task(param.translate)
        self.update_language(param.language)

        # Transcribe audio
        print("Transcribing audio...")
        start_time = time.time()
        transcription_result = await asyncio.to_thread(self.model.transcribe, audio, batch_size=self.batch_size)
        print("Transcription time: ", time.time() - start_time)

        # Diarize audio if requested
        diarization_result = None
        if param.diarize:
            print("Diarizing audio...")
            start_time = time.time()
            diarization_result = await asyncio.to_thread(self.diarize_model, audio)
            print("Diarization time: ", time.time() - start_time)

        # Word alignment
        start_time = time.time()
        transcription_result = await asyncio.to_thread(whisperx.align, transcription_result["segments"], self.align_model, self.metadata, audio, device=self.device)
        print("Alignment time: ", time.time() - start_time)

        # Assign speakers if diarization was performed
        if param.diarize:
            start_time = time.time()
            transcription_result = whisperx.assign_word_speakers(diarization_result, transcription_result)
            print("Speaker assignment time: ", time.time() - start_time)

        return transcription_result
    
    # For use with TTS functions
    def alignment(self, segments, audio):
        start_time = time.time()
        results = whisperx.align(segments, self.align_model, self.metadata, audio, device=self.device)
        print("Alignment time: ", time.time() - start_time)
        return results
    
    def set_task(self, translate: bool):
        if translate:
            self.model.task = 'translate'
        else:
            self.model.task = 'transcribe'
    
    def update_language(self, language: str):
        if language:
            self.model.language = language

    def clean_up(self, final=False):
        print("Cleaning up...")
        torch.cuda.empty_cache()  # Frequent cache clearing
        gc.collect()
        if final:
            for model_attr in ['model', 'align_model', 'diarize_model']:
                if hasattr(self, model_attr):
                    delattr(self, model_attr)
            torch.cuda.empty_cache()  # Additional clearing after deletion
            gc.collect()  # Final collection to clean up any residual references
