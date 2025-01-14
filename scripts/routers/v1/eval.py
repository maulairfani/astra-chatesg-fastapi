from fastapi import APIRouter, HTTPException

from scripts.schemas import FCMetricRequest
from scripts.services import FCMetric

router = APIRouter(prefix="/eval")
    
@router.post("/fc_metric")
async def fc_metric(request: FCMetricRequest):
    print(request)

    fc_metric = FCMetric()
    response = fc_metric.calculate(request)
    return response

    # try:
    #     chunk = service.create(chat_request)
    #     if chunk:
    #         response.status_code = 201
    #         return StreamingResponse(chunk, status_code=201, media_type="text/event-stream")
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error Generating chat response: {e}")