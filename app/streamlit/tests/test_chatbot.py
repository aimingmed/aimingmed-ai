import pytest
import streamlit as st
from unittest.mock import patch

# add app/streamlit to sys.path
import sys
sys.path.insert(0, "/Users/leehongkai/projects/aimingmed/aimingmed-ai/app/streamlit")

from unittest.mock import patch, MagicMock


def test_title():
    with patch("streamlit.title") as mock_title, \
         patch("streamlit.session_state", new_callable=MagicMock) as mock_session_state:
        import Chatbot
        st.session_state["messages"] = []
        mock_title.assert_called_once_with("💬 RAG AI for Medical Guideline")

def test_caption():
    with patch("streamlit.caption") as mock_caption, \
         patch("streamlit.session_state", new_callable=MagicMock) as mock_session_state:
        import Chatbot
        st.session_state["messages"] = []
        mock_caption.assert_called()

def test_chat_input():
    with patch("streamlit.chat_input", return_value="test_prompt") as mock_chat_input, \
         patch("streamlit.session_state", new_callable=MagicMock) as mock_session_state:
        import Chatbot
        st.session_state["messages"] = []
        mock_chat_input.assert_called_once()

def test_chat_message():
    with patch("streamlit.chat_message") as mock_chat_message, \
         patch("streamlit.session_state", new_callable=MagicMock) as mock_session_state:
        with patch("streamlit.chat_input", return_value="test_prompt"):
            import Chatbot
        st.session_state["messages"] = []
        mock_chat_message.assert_called()