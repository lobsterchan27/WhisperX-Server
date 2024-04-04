import io

from typing import Optional, Tuple, AsyncGenerator
from pydantic import BaseModel
from fastapi import Response
from fastapi.responses import StreamingResponse
from dataclasses import dataclass

class RequestParam(BaseModel):
    language: str = None
    text2speech: bool = False
    segment_audio: bool = False
    translate: bool = False
    get_video: bool = False

class TTSParam(BaseModel):
    text: str
    voice: str
    preset: str = 'ultra_fast'
    regenerate: Optional[str] = None
    seed: Optional[int] = None
    kv_cache: bool = True

class MultipartResponse(Response):
    media_type = "multipart/form-data"
    boundary = "--bulk-data-boundary"

    async def render(self, content: AsyncGenerator[Tuple[str, bytes], None]):
        for data_type, data in content:
            yield (
                f"{self.boundary}\r\n"
                f"Content-Type: {data_type}\r\n\r\n"
            ).encode()
            yield data
            yield "\r\n".encode()
        yield f"{self.boundary}--".encode()

    def __call__(self, content: AsyncGenerator[Tuple[str, bytes], None]):
        return StreamingResponse(
            self.render(content),
            headers={
                "Content-Type": f"multipart/form-data; boundary={self.boundary[2:]}"
            },
        )

@dataclass
class SavePath:
    audio: str
    json: str
    video: str = None