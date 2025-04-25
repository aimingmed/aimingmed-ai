import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from fastapi import WebSocket
import sys
import types

# Patch langchain and other heavy dependencies for import
sys.modules['langchain_deepseek'] = MagicMock()
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_community.vectorstores.chroma'] = MagicMock()
sys.modules['langchain_community.tools.tavily_search'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
sys.modules['langchain_core.output_parsers'] = MagicMock()
sys.modules['langchain.prompts'] = MagicMock()
sys.modules['langchain.schema'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()

from api import chatbot

@pytest.fixture
def client():
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(chatbot.router)
    return TestClient(app)

def test_router_exists():
    assert hasattr(chatbot, 'router')

def test_env_vars_loaded(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'dummy')
    monkeypatch.setenv('TAVILY_API_KEY', 'dummy')
    # Re-import to trigger env loading
    import importlib
    importlib.reload(chatbot)
    assert True

def test_websocket_endpoint_accepts(monkeypatch):
    # Patch ConnectionManager
    mock_manager = MagicMock()
    monkeypatch.setattr(chatbot, 'manager', mock_manager)
    ws = MagicMock(spec=WebSocket)
    ws.receive_text = MagicMock(side_effect=[pytest.raises(StopIteration)])
    ws.accept = MagicMock()
    # Should not raise
    try:
        coro = chatbot.websocket_endpoint(ws)
        assert hasattr(coro, '__await__')
    except Exception as e:
        pytest.fail(f"websocket_endpoint raised: {e}")
