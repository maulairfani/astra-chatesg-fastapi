from fastapi import APIRouter, HTTPException

from scripts.schemas import FCMetricRequest, RougeMetricRequest, RetrievalMetricRequest
from scripts.services import FCMetric, RougeMetric, PrecisionMetric, RecallMetric

router = APIRouter(prefix="/eval")
    
@router.post("/fc_metric")
async def fc_metric(request: FCMetricRequest):
    print(request)
    try:
        fc_metric = FCMetric()
        response = fc_metric.calculate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating fc_metric: {e}")

@router.post("/rouge_metric")
async def rouge_metric(request: RougeMetricRequest):
    print(request)
    try:
        fc_metric = RougeMetric()
        response = fc_metric.calculate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating rouge_metric: {e}")

@router.post("/precision_metric")
async def rouge_metric(request: RetrievalMetricRequest):
    print(request)
    try:
        fc_metric = PrecisionMetric()
        response = fc_metric.calculate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating precision_metric: {e}")

@router.post("/recall_metric")
async def rouge_metric(request: RetrievalMetricRequest):
    print(request)
    try:
        fc_metric = RecallMetric()
        response = fc_metric.calculate(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating recall_metric: {e}")