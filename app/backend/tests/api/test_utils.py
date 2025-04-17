import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

from fastapi import WebSocket

from api.utils import ConnectionManager

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# Test for ConnectionManager class
class TestConnectionManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = ConnectionManager()

    async def test_connect(self):
        mock_websocket = AsyncMock(spec=WebSocket)
        await self.manager.connect(mock_websocket)
        self.assertIn(mock_websocket, self.manager.active_connections)
        mock_websocket.accept.assert_awaited_once()

    async def test_disconnect(self):
        mock_websocket = MagicMock(spec=WebSocket)
        self.manager.active_connections.append(mock_websocket)
        self.manager.disconnect(mock_websocket)
        self.assertNotIn(mock_websocket, self.manager.active_connections)

    async def test_send_personal_message(self):
        mock_websocket = AsyncMock(spec=WebSocket)
        message = "Test message"
        await self.manager.send_personal_message(message, mock_websocket)
        mock_websocket.send_text.assert_awaited_once_with(message)

    async def test_broadcast(self):
        mock_websocket1 = AsyncMock(spec=WebSocket)
        mock_websocket2 = AsyncMock(spec=WebSocket)
        self.manager.active_connections = [mock_websocket1, mock_websocket2]
        message = "Broadcast message"
        await self.manager.broadcast(message)
        mock_websocket1.send_text.assert_awaited_once_with(
            '{"type": "message", "payload": "Broadcast message"}'
        )
        mock_websocket2.send_text.assert_awaited_once_with(
            '{"type": "message", "payload": "Broadcast message"}'
        )


if __name__ == "__main__":
    unittest.main()
