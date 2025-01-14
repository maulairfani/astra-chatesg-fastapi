from pydantic import BaseModel
from typing import List, Literal
from langchain_core.documents import Document

class MessageItem(BaseModel):
    type: Literal["text"]
    content: str
    role: Literal["system", "user", "assistant"]

class SourceItem(BaseModel):
    url: str | None = None
    pages: list[int] | None = None
    snippet: str | None = None
    related_indicators: list[str] | None = None

class ChatRequest(BaseModel):
    src: str
    bsid: str | None = None
    inputs: List[MessageItem]
    company: str = "PT Astra International Tbk"
    year: List[int] = 2022
    model: str
    retriever: Literal['similarity', 'indicator-cls', 'combination'] = 'indicator-cls'
    top_k: int | None = None

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
    mode: Literal['similarity', 'indicator-cls', 'combination'] = 'indicator-cls'
    top_k: int | None = None
    model: str | None = None

class RetrieverResponse(BaseModel):
    contents: List[Document] = []
    metadata: dict = {}