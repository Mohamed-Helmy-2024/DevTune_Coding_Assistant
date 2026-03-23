from fastapi import APIRouter, Depends , HTTPException
from fastapi.responses import JSONResponse
import logging

from .schemas.chat import ChatRequest ,HistoryRequest
from helpers.configs import Settings , get_settings
from controllers.ChatController import ChatController
from helpers.history import save_message , load_history

logger = logging.getLogger('uvicorn.error')

chat_router = APIRouter(
    prefix=f"/{get_settings().APP_NAME}/chat",
    tags=["DevTune"]
)

@chat_router.post("/complete")
async def complete_chat(request: ChatRequest, app_settings: Settings = Depends(get_settings)):
    try:
        controller = ChatController(
        session_id=request.session_id,
        username=request.username,
        utility_params=request.utility_params,
        chat_history=request.chat_history
        )

        ai_response = await controller.completion_router(request.prompt)

        # Save message + response
        await save_message(
            session_id=request.session_id,
            username=request.username,
            user_msg=request.prompt,
            ai_msg=ai_response
        )

        return JSONResponse({
            "session_id": request.session_id,
            "username": request.username,
            "response": ai_response
        })

    except Exception as e:
        logger.exception("pipeline error:")
        raise HTTPException(status_code=500, detail=str(e))
    

@chat_router.post("/history")
async def load_chat(request: HistoryRequest, app_settings: Settings = Depends(get_settings)):
    full_history = await load_history(request.session_id)
    cleaned=[]
    for pair in full_history:
        cleaned.append({"Human":pair[0].content})
        cleaned.append({"AI":pair[1].content})
    return JSONResponse({
            "session_id": request.session_id,
            "username": request.username,
            "history": cleaned
        })
    


    
