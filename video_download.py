import tempfile
import asyncio
import aiofiles
import re
import os
import math

from util import chunk_segments, generate_filtered_timestamps
from yt_dlp import YoutubeDL
from fastapi import UploadFile
from schema import RequestParam, SavePath


def sanitize_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename.replace(" ", "_")

# Save to disk the uploaded file
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

# Download the media from the given URL
def download_media(url, format, output_template, json=False):
    ydl_opts = {
        'format': format,
        'paths': {'home': '/download'},
        'restrictfilenames': True,
        'outtmpl': output_template,
        'writeinfojson': json,
        'quiet': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)

    if json:
        json_filename = filename.rsplit('.', 1)[0] + '.info.json'
        return filename, json_filename
    else:
        return filename, None

# Download and save the audio, video, and JSON files from the given URL
async def save_link(url, param: RequestParam) -> SavePath:
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

# Generate storyboards from the given video file
async def generate_storyboards(filename: str, param: RequestParam) -> str:
    print("Generating storyboards...")
    thumb_dir = os.path.join(os.path.dirname(filename), 'thumb')
    os.makedirs(thumb_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(thumb_dir, base_filename)

    # Command to get the original video's resolution
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', filename]
    ffprobe_process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    # Command to extract the timestamps of the scene changes or fixed intervals
    if param.scene_threshold and not param.fixed_interval:
        cmd = [
            'ffmpeg', '-i', filename, '-vf', 
            f'select=\'gt(scene\,{param.scene_threshold})\',scale=-1:320,showinfo',
            '-f', 'null', '-'
        ]
    elif param.fixed_interval:
        cmd = [
            'ffmpeg', '-i', filename, '-vf', 
            f'select=\'not(mod(t\,{param.fixed_interval}))\',scale=-1:320,showinfo',
            '-f', 'null', '-'
        ]
    ffmpeg_process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    # Read ffprobe output
    stdout, _ = await ffprobe_process.communicate()

    # Parse the output to find the original width and height
    width, height = map(int, stdout.decode().strip().split('x'))

    # Read ffmpeg output
    stdout, _ = await ffmpeg_process.communicate()

    # Generator to yield filtered timestamps based on the minimum interval
    filtered_timestamps = generate_filtered_timestamps(stdout, param.minimum_interval)

    # Chunk the timestamps into segments of length param.segment_length
    segmented_timestamps = chunk_segments(filtered_timestamps, param.segment_length)

    # Generate a storyboard for each chunk
    tasks = []
    for i, chunk in enumerate(segmented_timestamps):
        tasks.append(asyncio.create_task(generate_storyboard(chunk, width, height, filename, output_path, i)))
    results = await asyncio.gather(*tasks)
    return results

MAX_IMAGE_DIMENSION = 8192
# Generates each storyboard from the given timestamps
async def generate_storyboard(frames, width, height, filename, output_path, i = 0):
    aspect_ratio = width / height
    grid_rows, grid_cols, aspect_ratio = get_thumbnail_layout(len(frames), aspect_ratio)

    # Calculate the expected width and height of the final image with initial scale factor
    initial_scale_factor = 0.5
    expected_width = width * initial_scale_factor * grid_cols
    expected_height = height * initial_scale_factor * grid_rows

    # Calculate the scaling factor to ensure the final image size does not exceed the maximum limit
    if expected_width > MAX_IMAGE_DIMENSION or expected_height > MAX_IMAGE_DIMENSION:
        # Calculate scale factor based on the larger dimension
        scale_factor = (MAX_IMAGE_DIMENSION / max(expected_width, expected_height)) * initial_scale_factor
    else:
        scale_factor = initial_scale_factor

    print("Scale factor:", scale_factor)

    # Generate the thumbnail grid using the filtered timestamps
    select_filters = [f'between(t\,{timestamp-0.02}\,{timestamp+0.02})' for timestamp in frames]
    select_filter_str = '+'.join(select_filters)

    cmd = [
        'ffmpeg', '-y', '-i', filename, '-vf',
        f'select=\'{select_filter_str}\',tile={grid_cols}x{grid_rows},scale=iw*.5:-1',
        f'{output_path}_grid_{i}.webp'  # Include the chunk index in the filename
    ]
    process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await process.wait()
    return f'{output_path}_grid_{i}.webp'



# some issue with the algorithm. try to fix later(image is not very square at large numbers)
# goal is to calculate grid such that the overall image is 1:1 aspect ratio while individual cells
# minimize empty spaces
def get_thumbnail_layout(num_frames, aspect_ratio):
    # Calculate the desired grid size
    desired_grid_size = math.sqrt(num_frames * aspect_ratio)

    # Adjust the grid size to maintain the aspect ratio
    grid_rows = max(1, math.floor(desired_grid_size))

    # Adjust grid_rows and grid_cols until there are enough cells
    while True:
        grid_cols = max(1, math.ceil(num_frames / grid_rows))
        if grid_rows * grid_cols >= num_frames:
            break
        grid_rows += 1

    # If grid_cols is still too high, adjust grid_rows and grid_cols
    if grid_cols > math.ceil(num_frames ** 0.5):
        grid_cols = math.ceil(num_frames ** 0.5)
        grid_rows = math.ceil(num_frames / grid_cols)

    return grid_rows, grid_cols, aspect_ratio


if __name__ == "__main__":
    # The URL of the endpoint
    url = "http://localhost:8127/api/transcribe/url"

    # The file to upload
    youtube_link = "https://www.youtube.com/shorts/bhIPto46Ecg"

    # The RequestParam data
    param_data = {
        "language": "en",
        "text2speech": "False",
        "segment_length": "5",
        "translate": "False",
        "get_video": "True",
        "minimum_interval": "0.0",
        "scene_threshold": "0.02",
        "diarize": "True"
    }

    # The multipart/form-data payload
    payload = {
        "url": (None, youtube_link),
        "language": (None, param_data["language"]),
        "text2speech": (None, param_data["text2speech"]),
        "segment_length": (None, param_data["segment_length"]),
        "translate": (None, param_data["translate"]),
        "get_video": (None, param_data["get_video"]),
        "minimum_interval": (None, param_data["minimum_interval"]),
        "scene_threshold": (None, param_data["scene_threshold"]),
        "diarize": (None, param_data["diarize"])
    }

    import httpx
    timeout = httpx.Timeout(10, read=300)

    with httpx.stream('POST', url, files=payload, timeout=timeout) as response:
        for chunk in response.iter_raw():
            if chunk:
                print(chunk.decode(), flush=True)