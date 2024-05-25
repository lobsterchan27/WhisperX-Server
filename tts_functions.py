from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.audio import get_voices, load_audio, load_voice, get_voice_dir
from collections import deque

import edge_tts
from tortoise.api_fast import TextToSpeech as TextToSpeechFast, pad_or_truncate
from tortoise.api import TextToSpeech as TextToSpeechSlow

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

import librosa


from settings import (
    MODEL_DIR,
    OUTPUT_DIR,
    VOICES_DIRECTORY,
    DEVICE,
    COMPUTE_TYPE,
    TTS_TYPE,
    VOICE_CHUNK_DURATION_SIZE
)

last_voice = None
last_latents = None

voice_cache = {}
tts = None

def create_tts():
    if TTS_TYPE == 'fast':
        print("Using fast TTS")
        return TextToSpeechFast(use_deepspeed=True,
                            kv_cache=True,
                            half=True if COMPUTE_TYPE == 'float16' else False,
                            device=DEVICE)
    else:
        print("Using slow TTS")
        return TextToSpeechSlow(use_deepspeed=True,
                                minor_optimizations=True,
                            unsqueeze_sample_batches=False,
                            device=DEVICE)

    
def get_chunk_size( voice ):
    path = f'{get_voice_dir()}/{voice}/'
    if not os.path.isdir(path):
        return 1
    
    dataset_file = f'./training/{voice}/train.txt'
    if os.path.exists(dataset_file):
        return 0 # 0 will leverage using the LJspeech dataset for computing latents

    files = os.listdir(path)
    
    total = 0
    total_duration = 0

    for file in files:
        if file[-4:] != ".wav":
            continue

        metadata = torchaudio.info(f'{path}/{file}')
        duration = metadata.num_frames / metadata.sample_rate
        total_duration += duration
        total = total + 1


    # brain too fried to figure out a better way
    if VOICE_CHUNK_DURATION_SIZE == 0:
        result = int(total_duration / total) if total > 0 else 1
        return result
    result = int(total_duration / VOICE_CHUNK_DURATION_SIZE) if total_duration > 0 else 1
    # print(f"\n\nAutocalculated voice chunk duration size: {result}\n\n")
    return result

def load_or_generate_latents(tts, voice, directory: str):
    global last_voice, last_latents
    if voice != last_voice:
        save_path = f'{directory}/{voice}/{voice}.pth'
        if os.path.exists(save_path):
            last_latents = torch.load(save_path)
        else:
            last_latents = generate_latents(tts, voice, directory)
        last_voice = voice
    return last_latents
    

# def generate_latents(tts: TextToSpeechFast, voice, directory: str):
#     voices = get_voices([directory])
#     selected_voice = voice.split(',')
#     conds = []
#     for voice in selected_voice:
#         cond_paths = voices[voice]
#         for cond_path in cond_paths:
#             audio = load_voice(cond_path, 22050)
#     conditioning_latents = tts.get_conditioning_latents(conds)
#     save_path = f'{directory}/{voice}/{voice}.pth'
#     torch.save(conditioning_latents, save_path)
#     return conditioning_latents

def fetch_voice(tts, voice):
    global voice_cache
    cache_key = f'{voice}:{tts.autoregressive_model_hash[:8]}'
    if cache_key in voice_cache:
        return voice_cache[cache_key]

    voice_latent_chunks = get_chunk_size(voice)
    sample_voice = None
    
    if voice == 'random':
        voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
    else:
        voice_samples, conditioning_latents = load_voice(voice, model_hash=tts.autoregressive_model_hash)

    if voice_samples and len(voice_samples) > 0:
        if conditioning_latents is None:
            conditioning_latents = compute_latents(tts=tts, voice=voice, voice_samples=voice_samples, voice_latents_chunks=voice_latent_chunks)
                
        sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()
        voice_samples = None
        
    voice_cache[cache_key] = (voice_samples, conditioning_latents, sample_voice)
    return voice_cache[cache_key]


def compute_latents(tts, voice=None, voice_samples=None, voice_latents_chunks=0, original_ar=False, original_diffusion=False):

    if voice:
        load_from_dataset = voice_latents_chunks == 0

        if load_from_dataset:
            dataset_path = f'./training/{voice}/train.txt'
            if not os.path.exists(dataset_path):
                load_from_dataset = False
            else:
                with open(dataset_path, 'r', encoding="utf-8") as f:
                    lines = f.readlines()

                print("Leveraging dataset for computing latents")

                voice_samples = []
                max_length = 0
                for line in lines:
                    filename = f'./training/{voice}/{line.split("|")[0]}'
                    
                    waveform = load_audio(filename, 22050)
                    max_length = max(max_length, waveform.shape[-1])
                    voice_samples.append(waveform)

                for i in range(len(voice_samples)):
                    voice_samples[i] = pad_or_truncate(voice_samples[i], max_length)

                voice_latents_chunks = len(voice_samples)
                if voice_latents_chunks == 0:
                    print("Dataset is empty!")
                    load_from_dataset = True
        if not load_from_dataset:
            voice_samples, _ = load_voice(voice, load_latents=False)

    if voice_samples is None:
        return
    
    conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=False, slices=voice_latents_chunks, force_cpu=False, original_ar=original_ar, original_diffusion=original_diffusion)
    
    
    if len(conditioning_latents) == 4:
        conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
    
    outfile = f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth'
    torch.save(conditioning_latents, outfile)
    print(f'Saved voice latents: {outfile}')

    return conditioning_latents


def save_audio(tts, prompt, voice, resample=None):
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


def generate_tts(tts, prompt, voice):
    samplerate = 24000
    
    if '|' in prompt:
        print("Found the '|' character in your text, which I will use as a cue for where to split it up. If this was not"
              "your intent, please remove all '|' characters from the input.")
        prompts = prompt.split('|')
    else:
        prompts = split_and_recombine_text(prompt)
    
    _, conditioning_latents, _ = fetch_voice(tts, voice)
    # conditioning_latents = load_or_generate_latents(tts, voice, VOICES_DIRECTORY)

    settings = {'temperature': .4, 'length_penalty': 5.0, 'repetition_penalty': 2.0,
                'top_p': .8,
                'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
    
    presets = {
        'ultra_fast': {'num_autoregressive_samples': 8, 'diffusion_iterations': 30, 'cond_free': False},
        'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
        'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
        'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
        'custom': {'num_autoregressive_samples': 2, 'diffusion_iterations': 100, 'cond_free': True}
    }

    all_parts = []
    overall_time = time()
    for j, prompt in enumerate(prompts):
        print(f"Generating audio for prompt : {prompt}")
        start_time = time()
        if isinstance(tts, TextToSpeechFast):
            gen = tts.tts(prompt, voice_samples=None, conditioning_latents=conditioning_latents)
            
        if isinstance(tts, TextToSpeechSlow):
            settings = {'temperature': 0.4,
                        'top_p': 0.8,
                        'diffusion_temperature': 1.0,
                        'length_penalty': 6.0,
                        'repetition_penalty': 2.0,
                        'cond_free_k': 2.0,
                        'num_autoregressive_samples': 2,
                        'sample_batch_size': 2,
                        'diffusion_iterations': 100,
                        'voice_samples': None,
                        'k': 1,
                        'diffusion_sampler': 'DDIM',
                        'breathing_room': 8,
                        'half_p': False,
                        'cond_free': True,
                        'cvvp_amount': 0}
            gen = tts.tts(prompt,
                          half_p=True if COMPUTE_TYPE == 'float16' else False ,
                          voice_samples=None,
                          conditioning_latents=conditioning_latents,
                          temperature=0.4,
                          num_autoregressive_samples=2,
                          sample_batch_size=2,
                          diffusion_iterations=100,
                          length_penalty=8.0,
                          repetition_penalty=3.0,
                          cvvp_amount=0.0,
                          top_p=0.8,
                          diffusion_temperature=1.0,
                          diffusion_sampler='DDIM')
            
        end_time = time()
        audio = gen.squeeze().cpu()

        all_parts.append(audio)

        print("Time taken to generate the audio: ", end_time - start_time, "seconds")
        print("RTF: ", (end_time - start_time) / (audio.shape[0] / samplerate))
    full_audio = (torch.cat(all_parts, dim=-1)).numpy()
    duration = full_audio.shape[0] / samplerate  # Assuming a sample rate of 24,000 Hz
    print("Length of the audio: ", duration, "seconds")
    print("Total time taken to generate the audio: ", time() - overall_time, "seconds")

    return full_audio, duration, samplerate  # Return both the audio array and its duration

def generate_tts_stream(tts,
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

async def get_edge_tts(prompt, voice):
    output_file = 'temp.wav'
    communicate = edge_tts.Communicate(prompt, voice)
    await communicate.save(output_file)
    audio_data, sample_rate = sf.read(output_file)
    audio_data = np.squeeze(audio_data)
    duration = audio_data.shape[0] / sample_rate
    os.remove(output_file)
    return audio_data, sample_rate, duration


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