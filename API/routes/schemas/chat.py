from pydantic import BaseModel
from datetime import datetime

class ChatRequest(BaseModel):
    username: str = "anonymous"
    session_id: str = None
        
    prompt: str
    
    chat_history: list = []
    utility_params: dict = {"completion_type": "chat"}


class HistoryRequest(BaseModel):
    username: str = "anonymous"
    session_id: str = None
    
