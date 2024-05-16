import re
import gc
import numpy as np
import torch

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
def prepare_for_align(audio):
    return audio.astype(np.float32) / 32768.0

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()