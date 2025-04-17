import json
import os

from decouple import config
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain_deepseek import ChatDeepSeek

from .utils import ConnectionManager

router = APIRouter()

# Load environment variables
os.environ["DEEPSEEK_API_KEY"] = config("DEEPSEEK_API_KEY", cast=str)
os.environ["TAVILY_API_KEY"] = config("TAVILY_API_KEY", cast=str)

# Initialize the DeepSeek chat model
llm_chat = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize the connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            try:
                data_json = json.loads(data)
                if (
                    isinstance(data_json, list)
                    and len(data_json) > 0
                    and "content" in data_json[0]
                ):
                    async for chunk in llm_chat.astream(data_json[0]["content"]):
                        await manager.send_personal_message(
                            json.dumps({"type": "message", "payload": chunk.content}),
                            websocket,
                        )
                else:
                    await manager.send_personal_message(
                        "Invalid message format", websocket
                    )

            except json.JSONDecodeError:
                await manager.broadcast("Invalid JSON message")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client disconnected")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast("Client disconnected")
