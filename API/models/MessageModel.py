from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from helpers.database import Base

class MessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    username = Column(String, index=True)
    user_message = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
