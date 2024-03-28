import requests
import os
import torch
import torchaudio
from tortoise import api
from tortoise.utils.audio import load_voice



import os
from time import time

import torch

from tortoise.api_fast import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import load_audio, load_voice, get_voices
from schema import TTSParam
import sounddevice as sd
import queue
import threading
from time import sleep
from scipy.io.wavfile import write
import numpy as np
import pyaudio

def play_save_audio(audio_stream, sample_width, channels, sample_rate):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open PyAudio stream for playing audio
    play_stream = p.open(format=p.get_format_from_width(sample_width),
                         channels=channels,
                         rate=sample_rate,
                         output=True)

    # Reset the stream position to the beginning
    audio_stream.seek(0)

    # Read and play/save the audio data
    data_chunks = []
    data = audio_stream.read(1024)
    while data:
        # Play the audio data
        play_stream.write(data)

        # Append the data chunk to a list
        data_chunks.append(data)

        # Read the next chunk of data
        data = audio_stream.read(1024)

    # Stop the PyAudio stream
    play_stream.stop_stream()
    play_stream.close()

    # Terminate the PyAudio object
    p.terminate()

    # Concatenate all data chunks into a single NumPy array
    all_audio_data = b''.join(data_chunks)
    all_audio_numpy = np.frombuffer(all_audio_data, dtype=np.float32)

    # Write the audio data to a WAV file using scipy.io.wavfile.write
    write('output.wav', sample_rate, all_audio_numpy)


def tts_stream(TTSParam: TTSParam):

    models_dir = os.getenv("MODEL_DIRECTORY")
    deepspeed = False
    half = False

    extra_voice_dirs = [os.path.join(os.getcwd(), 'voices')]

    if torch.backends.mps.is_available():
        deepspeed = False
    tts = TextToSpeech(models_dir=models_dir, use_deepspeed=deepspeed, kv_cache=True, half=half)

    voice = TTSParam.voice

    # Process text
    input_text = TTSParam.text
    if '|' in input_text:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        texts = text.split('|')
    else:
        texts = split_and_recombine_text(input_text)

    audio_stream = io.BytesIO()
    # playback_thread = threading.Thread(target=play_audio, args=(audio_stream,))
    # playback_thread.start()

    voice_samples, _ = load_voice(voice, extra_voice_dirs)

    for j, text in enumerate(texts):
        audio_generator = tts.tts_stream(text, voice_samples=voice_samples, use_deterministic_seed=seed)
        for wav_chunk in audio_generator:
            print("chunk")
            audio_stream.write(wav_chunk.cpu().numpy().tobytes())

    # Reset the stream position to the beginning
    audio_stream.seek(0)

    return audio_stream

seed = 42
import io
from pydub import AudioSegment
from pydub.playback import play
if __name__ == "__main__":
    prompt = "Hello, my name is Tom. What is your name?"
    ttsparam = TTSParam(text=prompt, voice="tom", preset="ultra_fast", regenerate=None, seed=42, kv_cache=True)

    sample_width = 4  # Assuming 32-bit float
    channels = 1  # Mono audio
    sample_rate = 24000  # Sample rate of 24 kHz
    stream = tts_stream(ttsparam)
    play_save_audio(stream, sample_width, channels, sample_rate)