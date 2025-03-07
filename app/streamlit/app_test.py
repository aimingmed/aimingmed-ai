from unittest.mock import patch
from streamlit.testing.v1 import AppTest



@patch("langchain.llms.OpenAI.__call__")
def test_Langchain_Quickstart(langchain_llm):
    at = AppTest.from_file("pages/3_Langchain_Quickstart.py").run()
    assert at.info[0].value == "Please add your OpenAI API key to continue."

    RESPONSE = "1. The best way to learn how to code is by practicing..."
    langchain_llm.return_value = RESPONSE
    at.sidebar.text_input[0].set_value("sk-...")
    at.button[0].set_value(True).run()
    print(at)
    assert at.info[0].value == RESPONSE

