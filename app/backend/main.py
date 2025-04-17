import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import chatbot, ping

log = logging.getLogger("uvicorn")

origins = ["http://localhost:8004"]


def create_application() -> FastAPI:
    application = FastAPI()
    application.include_router(ping.router, tags=["ping"])
    application.include_router(chatbot.router, tags=["chatbot"])
    return application


app = create_application()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
