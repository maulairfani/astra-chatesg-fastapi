from pydantic import BaseModel
from typing import List, Literal
from datetime import datetime

class MessageItem(BaseModel):
    type: Literal["text"]
    content: str
    role: Literal["user", "assistant"]

class SessionDocument(BaseModel):
    version_no: int | None = 1
    src: str | None = None
    uid: str | None = None
    bsid: str | None = None
    is_deleted: bool | None = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    title: str | None = None
    suggestions: List | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

class ChatDocument(BaseModel):
    version_no: int = 1
    src: str
    uid: str | None = None
    bsid: str | None = None  
    bcid: str | None = None  
    is_deleted: bool | None = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    inputs: List[MessageItem] | None = None
    outputs: List[MessageItem] | None = None
    model: str | None = None
    start_time: datetime | None = None
    latency_ms: float | None = None  
    first_token_ms: float | None = None 
    report: str | None = None  
    sources: List | None = None
    metadata: dict | None = None 
    created_at: datetime | None = None
    updated_at: datetime | None = None
    thumbs_up: bool | None = None