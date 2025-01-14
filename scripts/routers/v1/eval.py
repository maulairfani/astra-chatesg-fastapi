from fastapi import APIRouter, HTTPException

from scripts.schemas import FCMetricRequest, RougeMetricRequest
from scripts.services import FCMetric, RougeMetric

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