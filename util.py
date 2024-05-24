import re
import gc
import numpy as np
import torch
import librosa

def chunk_segments(segments, segment_length, value_func=None, transform_func=None):
    current_chunk = []
    interval = segment_length

    for segment in segments:
        if (value_func(segment) if value_func else segment) >= interval:
            yield current_chunk
            current_chunk = []
            interval += segment_length
        current_chunk.append(transform_func(segment) if transform_func else segment)

    # Don't forget to yield the last chunk if it's not empty
    if current_chunk:
        yield current_chunk

def generate_filtered_timestamps(stdout, minimum_interval):
    prev_timestamp = None
    for m in re.finditer(r'pts_time:([0-9\.]+)', stdout.decode()):
        timestamp = float(m.group(1))
        if prev_timestamp is None or timestamp - prev_timestamp >= minimum_interval:
            yield timestamp
            prev_timestamp = timestamp

# Prepare rvc output for use with alignment
def prepare_audio(audio, sr, target_sr):
    # Convert the audio data to floating-point format if it's not already
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        print("Original audio data type: " + str(audio.dtype))
        print("Converting audio data to float32")
        audio = audio.astype(np.float32) / 32768.0

    # Resample the audio to target_sr
    resampled_audio = librosa.resample(audio,orig_sr=sr,target_sr=target_sr)

    # Convert to mono (if it's not already)
    if len(resampled_audio.shape) > 1:
        print("Converting audio to mono")
        resampled_audio = np.mean(resampled_audio, axis=0)

    # Normalize to the range -1.0 to 1.0 and convert to float32
    resampled_audio = resampled_audio.astype(np.float32) / np.max(np.abs(resampled_audio))

    return resampled_audio

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()