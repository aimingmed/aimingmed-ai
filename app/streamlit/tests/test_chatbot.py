import pytest
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot

import sys
sys.path.append(".")
import streamlit as st
from unittest.mock import patch
from Chatbot import CHAT_MODEL_PROVIDER, INPUT_CHROMADB_LOCAL, COLLECTION_NAME, cot_template, answer_template

@pytest.fixture(autouse=True)
def mock_session_state():
    with patch.object(st, "session_state", {"messages": []}):
        yield

def test_prompt_templates():
    # Test that the prompt templates are correctly formatted
    assert "documents_text" in cot_template
    assert "question" in cot_template
    assert "cot" in answer_template
    assert "question" in answer_template

def test_chromadb_connection():
    # Test that the ChromaDB client is initialized correctly
    chroma_client = chromadb.PersistentClient(path=INPUT_CHROMADB_LOCAL)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    assert collection is not None

@pytest.mark.skipif(CHAT_MODEL_PROVIDER not in ["deepseek", "gemini", "moonshot"], reason="requires a valid CHAT_MODEL_PROVIDER")
def test_llm_initialization():
    # Test that the correct LLM is initialized based on the CHAT_MODEL_PROVIDER environment variable
    if CHAT_MODEL_PROVIDER == "deepseek":
        llm = ChatDeepSeek(model="deepseek-chat")
        assert isinstance(llm, ChatDeepSeek)
    elif CHAT_MODEL_PROVIDER == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        assert isinstance(llm, ChatGoogleGenerativeAI)
    elif CHAT_MODEL_PROVIDER == "moonshot":
        llm = Moonshot(model="moonshot-v1-128k")
        assert isinstance(llm, Moonshot)
        llm = Moonshot(model="moonshot-v1-128k")
        assert isinstance(llm, Moonshot)