from typing import Optional
from pydantic import BaseModel

class RequestParam(BaseModel):
    language: str = None
    text2speech: Optional[bool] = False
    segment_audio: Optional[bool] = False
    translate: Optional[bool] = False
    get_video: Optional[bool] = False