from decouple import config
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT, 
    CONCISENESS_PROMPT, 
    HALLUCINATION_PROMPT
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot

GEMINI_API_KEY = config("GOOGLE_API_KEY", cast=str)
DEEKSEEK_API_KEY = config("DEEKSEEK_API_KEY", cast=str)
MOONSHOT_API_KEY = config("MOONSHOT_API_KEY", cast=str)

# correctness
gemini_evaluator_correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=GEMINI_API_KEY,
                temperature=0.5,
            ),
    )

deepseek_evaluator_correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.5,
                api_key=DEEKSEEK_API_KEY
            ),
    )

moonshot_evaluator_correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    judge=Moonshot(
                model="moonshot-v1-128k", 
                temperature=0.5,
                api_key=MOONSHOT_API_KEY
            ),
    )

# conciseness
gemini_evaluator_conciseness = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    judge=ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=GEMINI_API_KEY,
                temperature=0.5,
            ),
    )

deepseek_evaluator_conciseness = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    judge=ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.5,
                api_key=DEEKSEEK_API_KEY
            ),
    )

moonshot_evaluator_conciseness = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    judge=Moonshot(
                model="moonshot-v1-128k", 
                temperature=0.5,
                api_key=MOONSHOT_API_KEY
            ),
    )

# hallucination
gemini_evaluator_hallucination = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    judge=ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=GEMINI_API_KEY,
                temperature=0.5,
            ),
    )

deepseek_evaluator_hallucination = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    judge=ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.5,
                api_key=DEEKSEEK_API_KEY
            ),
    )

moonshot_evaluator_hallucination = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    judge=Moonshot(
                model="moonshot-v1-128k", 
                temperature=0.5,
                api_key=MOONSHOT_API_KEY
            ),
    )

