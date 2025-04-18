import pytest
import json
import websockets


@pytest.mark.asyncio
async def test_chatbot_integration():
    # Send a request to the chatbot endpoint
    url = "ws://backend-aimingmedai:80/ws"
    data = [{"content": "Hello"}]
    try:
        async with websockets.connect(url) as websocket:
            await websocket.send(json.dumps(data))
            response = await websocket.recv()
            assert response is not None
            try:
                response_json = json.loads(response)
                assert "type" in response_json
                assert "payload" in response_json
                assert response_json["payload"] == ""
            except json.JSONDecodeError:
                assert False, "Invalid JSON response"
    except Exception as e:
        pytest.fail(f"Request failed: {e}")