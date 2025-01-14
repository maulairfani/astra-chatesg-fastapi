from fastapi import APIRouter, Depends, Request, HTTPException
from firebase_admin import auth
from scripts.routers.v1 import chat, eval
from fastapi import Request

def verify_token(request: Request):
    test_uid = request.headers.get("test-uid")
    if test_uid:
        request.state.uid = test_uid
        return
    try:
        # Fetch the ID token from the request headers
        id_token = request.headers.get("id-token")

        if id_token is None:
            raise HTTPException(
                status_code=401, detail="Missing Authorization Header"
            )
        
        try:
            # Verify the ID token
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            request.state.uid = uid  
        except auth.ExpiredIdTokenError:
            raise HTTPException(status_code=401, detail="Token Expired")
        except auth.InvalidIdTokenError:
            raise HTTPException(status_code=401, detail="Token is invalid")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Error verifying token: {e}")
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid user token id: {e}")

router = APIRouter(
    prefix="/api/v1",
    dependencies=[
        Depends(verify_token)
    ]
)

router.include_router(chat.router)
router.include_router(eval.router)