import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocket, WebSocketDisconnect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.chatbot import llm_chat, manager, websocket_endpoint
