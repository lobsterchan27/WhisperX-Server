import io
import os
import json
import aiofiles
import mimetypes

from datetime import datetime
from typing import Optional, Tuple, AsyncGenerator, Dict, Any
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dataclasses import dataclass

class RequestParam(BaseModel):
    language: str = 'en'
    text2speech: bool = False
    segment_length: int = 30
    scene_threshold: Optional[float] = 0.02
    minimum_interval: float = 0
    fixed_interval: Optional[float] = None
    translate: bool = False
    get_video: bool = False
    diarize: Optional[bool] = False

class TTSRequest(BaseModel):
    prompt: str
    voice: str = 'reference'
    #below are not implemented yet.
    sample_rate: int = 24000
    preset: str = 'ultra_fast'
    regenerate: Optional[str] = None
    seed: Optional[int] = None
    kv_cache: bool = True

class MultipartResponse:
    media_type = "multipart/form-data"
    boundary = "bulk-data-boundary"

    async def body_iterator(self, content: AsyncGenerator[Tuple[str, Any, Optional[str]], None]):
        counter = 1
        async for data_type, data, *name_parts in content:
            name = name_parts[0] if name_parts else None
            file_data = None
            # Check if data is a file path
            if isinstance(data, str) and os.path.isfile(data):
                name = os.path.basename(data)
                async with aiofiles.open(data, 'rb') as f:
                    file_data = await f.read()
            # Generate a name if none was provided
            if not name:
                ext = mimetypes.guess_extension(data_type) or ".bin"
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                name = f"{timestamp}{ext}"


            if data_type == "application/json":
                yield (
                    f"--{self.boundary}\r\n"
                    f"Content-Disposition: form-data; name=\"{name}\"\r\n"
                    f"Content-Type: {data_type}\r\n\r\n"
                    f"{json.dumps(data)}\r\n"
                ).encode()
            else:
                if not file_data:
                    # Assume data is direct file data
                    file_data = data
                yield f"--{self.boundary}\r\n".encode()
                yield f"Content-Disposition: form-data; name=\"{counter}\"; filename=\"{name}\"\r\n".encode()
                yield f"Content-Type: {data_type}\r\n\r\n".encode()
                yield file_data
                yield "\r\n".encode()
            counter += 1
        yield f"--{self.boundary}--\r\n".encode()


    def __call__(self, content: AsyncGenerator[Tuple[str, Any], None], headers: Dict[str, str]):
        return StreamingResponse(
            self.body_iterator(content),
            media_type=f"{self.media_type}; boundary={self.boundary}",
            headers=headers
        )


@dataclass
class SavePath:
    basename: str
    audio: str
    json: str
    video: str = None