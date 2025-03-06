import datetime
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice


# See https://github.com/openai/openai-python/issues/715#issuecomment-1809203346
def create_chat_completion(response: str, role: str = "assistant") -> ChatCompletion:
    return ChatCompletion(
        id="foo",
        model="gpt-3.5-turbo",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=response,
                    role=role,
                ),
            )
        ],
        created=int(datetime.datetime.now().timestamp()),
    )


# @patch("langchain_deepseek.ChatDeepSeek.__call__")
# @patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
# @patch("langchain_community.llms.moonshot.Moonshot.__call__")
# def test_Chatbot(moonshot_llm, gemini_llm, deepseek_llm):
#     at = AppTest.from_file("Chatbot.py").run()
#     assert not at.exception
    
#     QUERY = "What is the best treatment for hypertension?"
#     RESPONSE = "The best treatment for hypertension is..."
    
#     deepseek_llm.return_value.content = RESPONSE
#     gemini_llm.return_value.content = RESPONSE
#     moonshot_llm.return_value = RESPONSE
    
#     at.chat_input[0].set_value(QUERY).run()
    
#     assert any(mock.called for mock in [deepseek_llm, gemini_llm, moonshot_llm])
#     assert at.chat_message[1].markdown[0].value == QUERY
#     assert at.chat_message[2].markdown[0].value == RESPONSE
#     assert at.chat_message[2].avatar == "assistant"
#     assert not at.exception


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
