import os
from decouple import config

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.llms.moonshot import Moonshot

from pydantic import BaseModel, Field

from prompts_library import CORRECTNESS_PROMPT, FAITHFULNESS_PROMPT

os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY", cast=str)
os.environ["DEEPSEEK_API_KEY"] = config("DEEPSEEK_API_KEY", cast=str)
os.environ["MOONSHOT_API_KEY"] = config("MOONSHOT_API_KEY", cast=str)


# Define output schema for the evaluation
class CorrectnessGrade(BaseModel):
    score: int = Field(description="Numerical score (1-5) indicating the correctness of the response.")

class FaithfulnessGrade(BaseModel):
    score: int = Field(description="Numerical score (1-5) indicating the faithfulness of the response.")



# Evaluators
def gemini_evaluator_correctness(outputs: dict, reference_outputs: dict) -> CorrectnessGrade:
    llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.5,
            )

    messages = [
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {"role": "user", "content": f"""Ground Truth answer: {reference_outputs["answer"]};
        Student's Answer: {outputs['response']}
        """}
    ]

    response = llm.invoke(messages)

    return CorrectnessGrade(score=int(response.content)).score


def deepseek_evaluator_correctness(outputs: dict, reference_outputs: dict) -> CorrectnessGrade:
    llm = ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.5,
            )

    messages = [
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {"role": "user", "content": f"""Ground Truth answer: {reference_outputs["answer"]};
        Student's Answer: {outputs['response']}
        """}
    ]

    response = llm.invoke(messages)

    return CorrectnessGrade(score=int(response.content)).score


def moonshot_evaluator_correctness(outputs: dict, reference_outputs: dict) -> CorrectnessGrade:
    llm = Moonshot(
                model="moonshot-v1-128k",
                temperature=0.5,
            )

    messages = [
        {"role": "system", "content": CORRECTNESS_PROMPT},
        {"role": "user", "content": f"""Ground Truth answer: {reference_outputs["answer"]};
        Student's Answer: {outputs['response']}
        """}
    ]

    response = llm.invoke(messages)

    try:
        return CorrectnessGrade(score=int(response)).score
    except ValueError:
        score_str = response.split(":")[1].strip()
        return CorrectnessGrade(score=int(score_str)).score
    

def gemini_evaluator_faithfulness(outputs: dict, reference_outputs: dict) -> FaithfulnessGrade:
    llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro", 
                temperature=0.5,
            )

    messages = [
        {"role": "system", "content": FAITHFULNESS_PROMPT},
        {"role": "user", "content": f"""Context: {reference_outputs["answer"]};
        Output: {outputs['response']}
        """}
    ]

    response = llm.invoke(messages)

    return FaithfulnessGrade(score=int(response.content)).score


def deepseek_evaluator_faithfulness(outputs: dict, reference_outputs: dict) -> FaithfulnessGrade:
    llm = ChatDeepSeek(
                model="deepseek-chat", 
                temperature=0.5,
            )

    messages = [
        {"role": "system", "content": FAITHFULNESS_PROMPT},
        {"role": "user", "content": f"""Context: {reference_outputs["answer"]};
        Output: {outputs['response']}
        """}
    ]

    response = llm.invoke(messages)

    return FaithfulnessGrade(score=int(response.content)).score


def moonshot_evaluator_faithfulness(outputs: dict, reference_outputs: dict) -> FaithfulnessGrade:
    llm = Moonshot(
                model="moonshot-v1-128k",
                temperature=0.5,
            )

    messages = [
        {"role": "system", "content": FAITHFULNESS_PROMPT},
        {"role": "user", "content": f"""Context: {reference_outputs["answer"]};
        Output: {outputs['response']}
        """}
    ]

    response = llm.invoke(messages)

    try:
        return FaithfulnessGrade(score=int(response)).score
    except ValueError:
        score_str = response.split(":")[1].strip()
        return FaithfulnessGrade(score=int(score_str)).score

