from tortoise import api_fast
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import get_voices, load_audio

import torch
import torchaudio
import os
import pprint
import pyaudio
import numpy as np

from time import time
from scipy.io.wavfile import write

from settings import MODEL_DIR, OUTPUT_DIR, VOICES_DIRECTORY

last_voice = None
last_latents = None


def create_tts():
    return api_fast.TextToSpeech(models_dir=MODEL_DIR, use_deepspeed=False, kv_cache=True, half=False, device='cuda')


def load_or_generate_latents(tts: api_fast.TextToSpeech, voice, directory: str):
    global last_voice, last_latents
    if voice != last_voice:
        save_path = f'{directory}/{voice}/{voice}.pth'
        if os.path.exists(save_path):
            last_latents = torch.load(save_path)
        else:
            last_latents = generate_latents(tts, voice, directory)
        last_voice = voice
    return last_latents, save_path
    

def generate_latents(tts: api_fast.TextToSpeech, voice, directory: str):
    voices = get_voices([directory])
    print(voices)  # Add this line
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


def save_audio(tts: api_fast.TextToSpeech, prompt, voice, resample=None):
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


def generate_tts(tts: api_fast.TextToSpeech, prompt, voice):
    conditioning_latents, _ = load_or_generate_latents(tts, voice, VOICES_DIRECTORY)
    start_time = time()
    gen = tts.tts(prompt, voice_samples=None, conditioning_latents=conditioning_latents, use_deterministic_seed=None)
    end_time = time()
    audio = gen.squeeze(0).cpu()
    print("Time taken to generate the audio: ", end_time - start_time, "seconds")
    print("RTF: ", (end_time - start_time) / (audio.shape[1] / 24000))
    return audio

def generate_tts_stream(tts, prompt, voice, audio_stream):
    # Process text
    if '|' in prompt:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        prompts = prompt.split('|')
    else:
        prompts = split_and_recombine_text(prompt)

    conditioning_latents, _ = load_or_generate_latents(tts, voice, VOICES_DIRECTORY)

    for prompt in prompts:
        gen = tts.tts_stream(prompt, voice_samples=None, conditioning_latents=conditioning_latents, use_deterministic_seed=None)
        for wav_chunk in gen:
            audio_stream.write(wav_chunk)

def play_audio(iobytes):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a new PyAudio stream outside the loop
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=24000,
                    output=True)

    chunks = []
    while True:
        # Check if the buffer is empty and print a debug message
        if buffer.tell() == 0:
            print("The buffer is currently empty.")

        # Read the next chunk from the buffer
        chunk = buffer.read()

        # If the chunk is empty, break the loop
        if not chunk:
            break

        # Convert the chunk to a numpy array and append it to the list
        chunk = np.frombuffer(chunk, dtype=np.float32)
        chunks.append(chunk)

        # Play the chunk
        stream.write(chunk.tobytes())

    # Close the stream after the loop
    stream.close()

    # Concatenate all chunks
    all_audio = np.concatenate(chunks)
    
    # Write the output to a WAV file
    write("output.wav", 24000, all_audio)

    # Terminate the PyAudio object
    p.terminate()

if __name__=='__main__':
    import io
    import threading
    tts = create_tts()
    prompt = "Hello, my name is Tortoise. I am a text-to-speech model."
    voice = "reference"
    buffer = io.BytesIO()
    playback_thread = threading.Thread(target=play_audio, args=(buffer,))
    playback_thread.start()
    generate_tts_stream(tts, prompt, voice, buffer)
    playback_thread.join()
    buffer.close()