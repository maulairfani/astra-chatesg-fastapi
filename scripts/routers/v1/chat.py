from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from scripts.services import ChatService, Retriever
from scripts.schemas import ChatRequest, RetrieverRequest
from scripts.utils import init_chain

router = APIRouter()

@router.post("/chat")
async def chat(request: Request, response: Response, chat_request: ChatRequest):
    print(chat_request)
    uid = request.state.uid
    service = ChatService(request.app.fsclient, uid, chat_request.model)

    try:
        chunk = service.create(chat_request)
        if chunk:
            response.status_code = 201
            return StreamingResponse(chunk, status_code=201, media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Generating chat response: {e}")
    
@router.post("/retrieval")
async def retrieval(request: Request, response: Response, retriever_request: RetrieverRequest):
    print(retriever_request)
    service = Retriever(request.app.fsclient)

    contexts = service.get_relevant_contents(retriever_request)
    return contexts

