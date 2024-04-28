# import requests
# import os
# import torch
# import torchaudio
# from tortoise import api
# from tortoise.utils.audio import load_voice



# import os
# from time import time

# import torch

# from tortoise.api_fast import TextToSpeech
# from tortoise.utils.text import split_and_recombine_text
# from tortoise.utils.audio import load_audio, load_voice, get_voices
# from schema import TTSParam
# import sounddevice as sd
# import queue
# import threading
# from time import sleep
# from scipy.io.wavfile import write
# import numpy as np
# import pyaudio

# def play_save_audio(audio_stream, sample_width, channels, sample_rate):
#     # Initialize PyAudio
#     p = pyaudio.PyAudio()

#     # Open PyAudio stream for playing audio
#     play_stream = p.open(format=p.get_format_from_width(sample_width),
#                          channels=channels,
#                          rate=sample_rate,
#                          output=True)

#     # Reset the stream position to the beginning
#     audio_stream.seek(0)

#     # Read and play/save the audio data
#     data_chunks = []
#     data = audio_stream.read(1024)
#     while data:
#         # Play the audio data
#         play_stream.write(data)

#         # Append the data chunk to a list
#         data_chunks.append(data)

#         # Read the next chunk of data
#         data = audio_stream.read(1024)

#     # Stop the PyAudio stream
#     play_stream.stop_stream()
#     play_stream.close()

#     # Terminate the PyAudio object
#     p.terminate()

#     # Concatenate all data chunks into a single NumPy array
#     all_audio_data = b''.join(data_chunks)
#     all_audio_numpy = np.frombuffer(all_audio_data, dtype=np.float32)

#     # Write the audio data to a WAV file using scipy.io.wavfile.write
#     write('output.wav', sample_rate, all_audio_numpy)


# seed = 42
# import io
# from pydub import AudioSegment
# from pydub.playback import play
# if __name__ == "__main__":
#     prompt = "Hello, my name is Tom. What is your name?"
#     ttsparam = TTSParam(text=prompt, voice="tom", preset="ultra_fast", regenerate=None, seed=42, kv_cache=True)

#     sample_width = 4  # Assuming 32-bit float
#     channels = 1  # Mono audio
#     sample_rate = 24000  # Sample rate of 24 kHz
#     stream = tts_stream(ttsparam)
#     play_save_audio(stream, sample_width, channels, sample_rate)

#     segments = transcript['segments']
#     chunked_segments = []
#     current_chunk = []
#     interval = param.segment_length

#     for segment in segments:
#         if segment['start'] >= interval:
#             chunked_segments.append(current_chunk)
#             current_chunk = []
#             interval += param.segment_length
#         current_chunk.append({key: segment[key] for key in segment if key != 'words'})

#     # Append the last chunk
#     if current_chunk:
#         chunked_segments.append([{key: segment[key] for key in segment if key != 'words'} for segment in current_chunk])

#     response = []
#     for i, (chunk, storyboard) in enumerate(zip(chunked_segments, storyboards)):
#         response.append({
#             "Storyboard": storyboard,
#             "Segments": chunk
#         })

import requests
import json

# url = "http://localhost:8127/api/text2speech"
url = "http://96.253.117.172:8000/api/extra/generate/stream"

payload = {
  "prompt": 
  '''
Ok what next?
So I have been a bit quiet on the blog \'whatever bro\'''',
  "temperature": 0.5,
  "top_p": 0.9,
  "max_length": 2
}

response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
for line in response.iter_lines():
    # filter out keep-alive new lines
    if line:
        decoded_line = line.decode('utf-8')
        if decoded_line.startswith('data:'):
            data = json.loads(decoded_line[5:])
            print(data)