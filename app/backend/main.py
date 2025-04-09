import logging

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware


from config import get_settings, Settings

from api import ping, chatbot

log = logging.getLogger("uvicorn")

origins = ["http://localhost:3000"]

def create_application() -> FastAPI:
    application = FastAPI()
    application.include_router(ping.router, tags=["ping"])
    application.include_router(
        chatbot.router, tags=["chatbot"])
    return application


app = create_application()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3100, reload=True)