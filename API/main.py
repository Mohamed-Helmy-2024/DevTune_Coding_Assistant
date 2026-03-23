import http
from fastapi import FastAPI
from routes import base, chat, rag
from models.MessageModel import MessageModel
from helpers.database import engine
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://127.0.0.1:8000",  # Django dev server
    "http://localhost:8000",
    "*",  # allow all (dev only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(MessageModel.metadata.create_all)
    print("✔ Database tables created.")

app.include_router(base.base_router)
app.include_router(chat.chat_router)
app.include_router(rag.rag_router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5555,
    )