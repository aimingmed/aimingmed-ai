import os
import sys
import json
from unittest.mock import AsyncMock, MagicMock
import unittest

from unittest.mock import AsyncMock
from fastapi import WebSocket, WebSocketDisconnect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.chatbot import websocket_endpoint, manager, llm_chat

