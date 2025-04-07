import logging

from fastapi import FastAPI, Depends

from config import get_settings, Settings

from api import ping, query

log = logging.getLogger("uvicorn")


def create_application() -> FastAPI:
    application = FastAPI()
    application.include_router(ping.router)
    application.include_router(
        query.router, prefix="/query", tags=["query"]
    )

    return application


app = create_application()


