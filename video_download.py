import tempfile
import asyncio
import aiofiles
import subprocess
import re
import os
import math

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from yt_dlp import YoutubeDL
from fastapi import UploadFile
from schema import RequestParam, SavePath

async def save_upload_file(upload_file: UploadFile) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            data = await upload_file.read()
            async with aiofiles.open(temp_file.name, 'wb') as f:
                await f.write(data)
            return temp_file.name
    except Exception as e:
        print(f"Failed to save upload file: {e}")
        return None

def sanitize_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename.replace(" ", "_")

def download_media(url, format, output_template, json=False):
    ydl_opts = {
        'format': format,
        'paths': {'home': '/download'},
        'restrictfilenames': True,
        'outtmpl': output_template,
        'writeinfojson': json,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)

    if json:
        json_filename = filename.rsplit('.', 1)[0] + '.info.json'
        return filename, json_filename
    else:
        return filename, None


async def save_link(executor: ThreadPoolExecutor, url, param: RequestParam) -> SavePath:
    """
    Download video, audio, and JSON from a given URL and save them in a folder.

    The folder and file structure is as follows:
    /download/title/title.audio
    /download/title/title_video.video (if param.get_video is True)
    /download/title/title.info.json

    The function submits tasks to download the audio and (optionally) video to the executor. It then waits for each task to complete and retrieves the results.

    If param.get_video is True, a video file is downloaded, otherwise, no video file is downloaded.

    Returns a SavePath object containing the paths to the downloaded audio, JSON, and (optionally) video files.
    """

    location = '%(title)s/%(title)s'

    audio_task = asyncio.create_task(asyncio.to_thread(download_media, url, 'bestaudio', f'{location}.%(ext)s', True))
    video_task = asyncio.create_task(asyncio.to_thread(download_media, url, 'bestvideo', f'{location}_video.%(ext)s')) if param.get_video else None

    # Wait for all tasks to complete
    audio_path, json_path = await audio_task
    video_path = (await video_task)[0] if video_task else None

    return SavePath(audio=audio_path, json=json_path, video=video_path)

# To be refactored
async def generate_storyboard(filename: str) -> str:
    thumb_dir = os.path.join(os.path.dirname(filename), 'thumb')
    os.makedirs(thumb_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(thumb_dir, base_filename)

    # Command to get the original video's resolution
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', filename]
    output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Parse the output to find the original width and height
    width, height = map(int, output.stdout.decode().strip().split('x'))
    aspect_ratio = width / height

    cmd = [
        'ffmpeg', '-i', filename, '-vf', 
        'select=\'gt(scene\,0.02)\',scale=-1:320,showinfo',
        '-f', 'null', '-'
    ]
    output = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


    # Parse the output to find the timestamps of the selected frames
    timestamps = [float(m.group(1)) for m in re.finditer(r'pts_time:([0-9\.]+)', output.stdout.decode())]

    # Filter the timestamps to keep only those where at least x seconds have passed since the last selected frame
    filtered_timestamps = []
    prev_timestamp = None
    for timestamp in timestamps:
        if prev_timestamp is None or timestamp - prev_timestamp >= 1:
            filtered_timestamps.append(timestamp)
            prev_timestamp = timestamp

    # Determine the grid dimensions
    num_frames = len(filtered_timestamps)
    grid_size = math.sqrt(num_frames * aspect_ratio)

    # Round grid_rows down and grid_cols up to ensure enough cells
    grid_rows = math.floor(grid_size)
    grid_cols = math.ceil(num_frames / grid_rows)

    # If not enough cells, increment grid_rows
    if grid_rows * grid_cols < num_frames:
        grid_rows += 1

    # Generate the thumbnail grid using the filtered timestamps
    cmd = [
        'ffmpeg', '-y', '-i', filename, '-vf',
        f'select=\'gte(t\,{filtered_timestamps[0]-0.02})*lte(t\,{filtered_timestamps[0]+0.02})\'+' + '+'.join(f'gte(t\,{timestamp-0.02})*lte(t\,{timestamp+0.02})' for timestamp in filtered_timestamps[1:]) +
        f',tile={grid_cols}x{grid_rows},scale=iw*.5:-1',
        f'{output_path}_grid.webp'
    ]
    subprocess.run(cmd, check=True)
    print(num_frames, grid_rows, grid_cols, aspect_ratio)
    return output_path


if __name__ == "__main__":
    # Run the save_link coroutine
    # result = asyncio.run(save_link("https://www.youtube.com/shorts/W2xxT3b-4H0", RequestParam(language='en', get_video=True)))
    # folder_name = os.path.basename(result)
    # joined = os.path.join(result, folder_name)
    # files = glob.glob(f"{joined}.*")
    # if files:
    #     asyncio.run(generate_storyboard(files[0]))

    import requests

    # The URL of the endpoint
    url = "http://localhost:8127/api/transcribe/file"

    # The file to upload
    file_path = "D:\Cool\WhisperX Server\download\Look_At_What_Happens_When_I_Heat_Treat_a_Metal_Lattice\Look_At_What_Happens_When_I_Heat_Treat_a_Metal_Lattice_audio.webm"

    # The RequestParam data
    param_data = {
        "language": "en",
        "text2speech": "False",
        "segment_audio": "False",
        "translate": "False",
        "get_video": "False"
    }

    # The multipart/form-data payload
    payload = {
        "file": ("filename", open(file_path, 'rb')),
        "language": (None, param_data["language"]),
        "text2speech": (None, param_data["text2speech"]),
        "segment_audio": (None, param_data["segment_audio"]),
        "translate": (None, param_data["translate"]),
        "get_video": (None, param_data["get_video"])
    }

    # Send the POST request
    response = requests.post(url, files=payload)

    # Print the response
    print(response.json())