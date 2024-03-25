import requests
import os
import torch
import torchaudio
from dotenv import load_dotenv
from tortoise import api
from tortoise.utils.audio import load_voice

load_dotenv()
filepath = "C:/Users/Lobby/Look At What Happens When I Heat Treat a Metal Lattice! [W2xxT3b-4H0].webm"

def whisperx_response(video_path):
    url = "http://127.0.0.1:8127/api/audio"
    params = {"file": video_path}
    try:
        response = requests.post(url, params=params)
        if response.status_code != 200:
            print(response.content)  # Print the response content if the status code is not 200
        data = response.json()
        segments = data['segments']
        language = data['language']
        return segments, language
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # segments, language = whisperx_response(filepath)
    # print("Segments:", segments)
    # print("Language:", language)
    model_path = os.getenv("Model_directory")
    tts = api.TextToSpeech(models_dir=model_path, use_deepspeed=False, kv_cache=True, half=False, device='cuda')
    voice = 'kitty'

    extra_voice_dirs = [os.path.join(os.getcwd(), 'voices')]

    voice_samples, conditioning_latents = load_voice(voice, extra_voice_dirs)
    torch.save(voice_samples, os.path.join('voices', f"{voice}_samples.pth"))
    text = "Oh my god, why can't I keep my lips off this cock"

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gen = tts.tts_with_preset(text, preset='ultra_fast', k=1, voice_samples=voice_samples, conditioning_latents=conditioning_latents)
    torchaudio.save(os.path.join(output_dir, f'{voice}.wav'), gen.squeeze(0).cpu(), 24000)