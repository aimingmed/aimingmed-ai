import pytest
import websockets

@pytest.mark.asyncio
async def test_websocket_connection():
    url = "ws://backend-aimingmedai:80/ws"
    try:
        async with websockets.connect(url):
            assert True  # If the connection is established, the test passes
    except Exception:
        assert False  # If any exception occurs, the test fails