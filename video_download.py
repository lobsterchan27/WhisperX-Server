import tempfile
from yt_dlp import YoutubeDL
from fastapi import UploadFile
import asyncio
import subprocess
import re
import os
from yt_dlp.postprocessor.ffmpeg import FFmpegPostProcessor

from schema import AudioParams

async def save_upload_file(upload_file: UploadFile) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await upload_file.read())
            return temp_file.name
    except Exception as e:
        print(f"Failed to save upload file: {e}")
        return None

def sanitize_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename.replace(" ", "_")


async def save_link(url, param: AudioParams) -> str:

    format = "bestaudio,bestvideo"

    ydl_opts = {
        'format': format,
        'paths': {'home': '/download'},
        'restrictfilenames': True,
        'outtmpl': '%(title)s/%(title)s.%(ext)s',
        'writeinfojson': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)

    filename = ydl.prepare_filename(info_dict)
    thumb_dir = os.path.join(os.path.dirname(filename), 'thumb')
    os.makedirs(thumb_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    cmd = [
        'ffmpeg', '-i', filename, '-vf', 
        'select=\'gt(scene\,0.01)\',showinfo,scale=-1:320,pad=iw+2:ih+2:1:1:magenta,drawtext=text=\'%{pts\:hms}\':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=10',
        '-f', 'null', '-'
    ]
    output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Parse the output to find the timestamps of the selected frames
    timestamps = [float(m.group(1)) for m in re.finditer(r'pts_time:([0-9\.]+)', output.stdout.decode())]

    # Filter the timestamps to keep only those where at least 2 seconds have passed since the last selected frame
    filtered_timestamps = []
    prev_timestamp = None
    for timestamp in timestamps:
        if prev_timestamp is None or timestamp - prev_timestamp >= 0:
            filtered_timestamps.append(timestamp)
            prev_timestamp = timestamp

    # Generate the thumbnail grid using the filtered timestamps
    cmd = [
        'ffmpeg', '-i', filename, '-vf', 
        f'select=\'eq(t\,{filtered_timestamps[0]})\'+' + '+'.join(f'eq(t\,{timestamp})' for timestamp in filtered_timestamps[1:]) + ',tile=5x5',
        f'{thumb_dir}/{base_filename}_grid.webp'
    ]
    subprocess.run(cmd, check=True)
    print(filename)
    print(os.path.basename(filename))
    return filename


if __name__ == "__main__":
    # Run the save_link coroutine
    result = asyncio.run(save_link("https://www.youtube.com/shorts/bhIPto46Ecg", AudioParams(language='en', get_video=True)))
    print(result)