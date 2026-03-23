from .database import AsyncSessionLocal
from models.MessageModel import MessageModel
from sqlalchemy import select
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

async def save_message(session_id, username, user_msg, ai_msg):
    async with AsyncSessionLocal() as session:
        chat = MessageModel(
            session_id=session_id,
            username=username,
            user_message=user_msg,
            ai_response=ai_msg,
            timestamp=datetime.utcnow()
        )
        session.add(chat)
        await session.commit()


async def load_history(session_id):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(MessageModel)
            .where(MessageModel.session_id == session_id)
            .order_by(MessageModel.timestamp)
        )
        rows = result.scalars().all()

        history = []
        for row in rows:
            rec=[]
            rec.append(HumanMessage(content=row.user_message if row.user_message is not None else ""))
            rec.append(AIMessage(content=row.ai_response if row.ai_response is not None else ""))
            history.append(rec)
        return history
