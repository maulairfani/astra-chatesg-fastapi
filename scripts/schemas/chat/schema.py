from pydantic import BaseModel
from typing import List, Literal
from langchain_core.documents import Document

class MessageItem(BaseModel):
    type: Literal["text"]
    content: str
    role: Literal["system", "user", "assistant"]

class SourceItem(BaseModel):
    url: str
    pages: list[int] | None = None
    snippet: str

class ChatRequest(BaseModel):
    src: str
    bsid: str | None = None
    inputs: List[MessageItem]
    company: str = "PT Astra International Tbk"
    year: List[int] = 2022
    model: str

class ChatResponse(BaseModel):
    done: bool = False
    tools: List[str] = []
    responses: List[str] = []
    sources: List[dict] | None = None
    bsid: str | None = None
    bcid: str | None = None
    metadata: dict | None = None

class RetrieverRequest(BaseModel):
    query: str
    filter: dict

class RetrieverResponse(BaseModel):
    contents: List[Document]