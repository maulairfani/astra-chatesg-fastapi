from pydantic import BaseModel
from typing import Literal, List

class FCMetricRequest(BaseModel):
    response: str
    reference: str
    model_name: str = 'gpt-4o-mini'
    mode: Literal['f1', 'recall', 'precision'] = 'recall'
    atomicity: Literal['high', 'low'] = 'high'
    coverage: Literal['high', 'low'] = 'high'

class FCMetricResponse(BaseModel):
    score: float
    response_claims: List[str]
    reference_claims: List[str]
    response_reference: List[bool]
    reference_response: List[bool]

class RougeMetricRequest(BaseModel):
    response: str
    reference: str
    rouge_type: Literal['rouge1', 'rouge2', 'rougeL'] = 'rougeL'
    measure_type: Literal['recall', 'precision', 'f1'] = 'recall'

class RougeMetricResponse(BaseModel):
    score: float