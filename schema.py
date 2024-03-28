from typing import Optional
from pydantic import BaseModel

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