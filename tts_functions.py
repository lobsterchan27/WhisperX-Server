from tortoise.api_fast import TextToSpeech
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import get_voices, load_audio
from collections import deque

import torch
import torchaudio
import os
import pyaudio
import numpy as np
import soundfile as sf

from io import BytesIO
from queue import Queue
from time import time
from scipy.io.wavfile import write


from settings import MODEL_DIR, OUTPUT_DIR, VOICES_DIRECTORY

last_voice = None
last_latents = None


def create_tts():
    return TextToSpeech(models_dir=MODEL_DIR,
                        use_deepspeed=False,
                        kv_cache=True,
                        half=False,
                        device='cuda')


def load_or_generate_latents(tts: TextToSpeech, voice, directory: str):
    global last_voice, last_latents
    if voice != last_voice:
        save_path = f'{directory}/{voice}/{voice}.pth'
        print(f"Loading latents from: {save_path}")  # Add this line
        if os.path.exists(save_path):
            last_latents = torch.load(save_path)
        else:
            last_latents = generate_latents(tts, voice, directory)
        last_voice = voice
    return last_latents
    

def generate_latents(tts: TextToSpeech, voice, directory: str):
    voices = get_voices([directory])
    selected_voice = voice.split(',')
    conds = []
    for voice in selected_voice:
        cond_paths = voices[voice]
        for cond_path in cond_paths:
            audio = load_audio(cond_path, 22050)
            conds.append(audio)
    conditioning_latents = tts.get_conditioning_latents(conds)
    save_path = f'{directory}/{voice}/{voice}.pth'
    torch.save(conditioning_latents, save_path)
    return conditioning_latents


def save_audio(tts: TextToSpeech, prompt, voice, resample=None):
    output_dir = os.path.join(OUTPUT_DIR, voice)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the audio
    audio = generate_tts(tts, prompt, voice)

    # Resample the audio if a resample rate is provided
    if resample is not None:
        audio = resample_audio(audio, resample)

    num = 1
    while os.path.exists(os.path.join(output_dir, f'{voice}_{num:03}.wav')):
        num += 1

    # Save the audio with the new filename
    torchaudio.save(os.path.join(output_dir, f'{voice}_{num:03}.wav'), audio, resample if resample else 24000)


def resample_audio(audio, resample: int):
    resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=resample)
    audio = resampler(audio)
    return audio


def generate_tts(tts: TextToSpeech, prompt, voice):
    conditioning_latents = load_or_generate_latents(tts, voice, VOICES_DIRECTORY)
    start_time = time()
    gen = tts.tts(prompt, voice_samples=None, conditioning_latents=conditioning_latents)
    end_time = time()
    audio = gen.squeeze().cpu().numpy()
    print("Time taken to generate the audio: ", end_time - start_time, "seconds")
    print("RTF: ", (end_time - start_time) / (audio.shape[0] / 24000))
    sf.write('debug1.wav', audio, 24000)
    return audio

def generate_tts_stream(tts: TextToSpeech,
                        prompt,
                        voice,
                        audio_queue: deque=None,
                        save_path=None,
                        event=None):
    # Process text
    if '|' in prompt:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        prompts = prompt.split('|')
    else:
        prompts = split_and_recombine_text(prompt)

    conditioning_latents = load_or_generate_latents(tts, voice, VOICES_DIRECTORY)

    # Initialize the audio data list if a save path is provided
    audio_data = [] if save_path is not None else None

    # Generate the audio for each prompt split by '|' or split_and_recombine_text
    for prompt in prompts:
        gen = tts.tts_stream(prompt, voice_samples=None, conditioning_latents=conditioning_latents)
        for wav_chunk in gen:
            cpu_chunk = wav_chunk.cpu()
            if save_path is not None:
                audio_data.append(cpu_chunk)
            if audio_queue is not None:
                audio_queue.extend(cpu_chunk.numpy().tobytes())
                # event for use with rvc
                if event is not None:
                    event.set()
    
    # Save the audio if a save path is provided
    if save_path is not None and audio_data:
        audio_data = torch.cat(audio_data, dim=-1)
        audio_data = audio_data.unsqueeze(0)
        torchaudio.save(save_path, audio_data, 24000)

# Local function to play audio
def play_audio(audio_queue: deque):
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=24000,
                    output=True)

    while True:
        if not audio_queue:  # If the deque is empty, break the loop
            break
        chunk = audio_queue.popleft()  # Get the next chunk
        stream.write(chunk)

    # Clean up PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()

def to_wav(audio, samplerate):
    byte_io = BytesIO()
    sf.write(byte_io, audio, samplerate, format='WAV')
    byte_io.seek(0)
    return byte_io.getvalue()

if __name__=='__main__':
    import io
    import threading
    import soundfile as sf
    from rvc_pipe.rvc_infer import rvc_convert

    tts = create_tts()
    prompt = "Hello, my name is Elizabeth. Saint take out the trash I don't have all day. Also when you're done with that I need you to sweep up the garage."
    voice = "reference"
    audio_queue = Queue()
    generate_thread = threading.Thread(target=generate_tts_stream, args=(tts, prompt, voice, audio_queue, 'elizabeth'))
    play_thread = threading.Thread(target=play_audio, args=(audio_queue,))

    generate_thread.start()
    play_thread.start()

    # Wait for both threads to finish
    generate_thread.join()
    play_thread.join()

    rvc_convert(model_path='models/rvc_models/FrierenFrierenv3_e150_s15000.pth',
                input_path='elizabeth.wav')

    