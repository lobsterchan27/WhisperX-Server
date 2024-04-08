import io
import json

from typing import Optional, Tuple, AsyncGenerator, Any
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from dataclasses import dataclass

class RequestParam(BaseModel):
    language: str = None
    text2speech: bool = False
    segment_length: int = 30
    minimum_interval: float = 0
    translate: bool = False
    get_video: bool = False
    scene_threshold: Optional[float] = 0.02
    fixed_interval: Optional[float] = None
    diarize: Optional[bool] = False

class TTSRequest(BaseModel):
    prompt: str
    voice: str
    #below are not implemented yet.
    sample_rate: int = 24000
    preset: str = 'ultra_fast'
    regenerate: Optional[str] = None
    seed: Optional[int] = None
    kv_cache: bool = True

class MultipartResponse:
    media_type = "multipart/form-data"
    boundary = "--bulk-data-boundary"

    async def body_iterator(self, content: AsyncGenerator[Tuple[str, Any], None]):
        async for data_type, data in content:
            yield (
                f"{self.boundary}\r\n"
                f"Content-Type: {data_type}\r\n\r\n"
            ).encode()
            if data_type == "application/json":
                yield json.dumps(data).encode()
            else:
                #testing/debug load actual binary data later
                yield data.encode()
            yield "\r\n".encode()
        yield f"{self.boundary}--".encode()

    def __call__(self, content: AsyncGenerator[Tuple[str, Any], None]):
        return StreamingResponse(
            self.body_iterator(content),
            headers={
                "Content-Type": f"multipart/form-data; boundary={self.boundary[2:]}"
            },
        )


@dataclass
class SavePath:
    audio: str
    json: str
    video: str = None