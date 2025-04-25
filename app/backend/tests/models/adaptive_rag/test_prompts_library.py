import pytest
from models.adaptive_rag import prompts_library

def test_prompts_are_strings():
    assert isinstance(prompts_library.system_router, str)
    assert isinstance(prompts_library.system_retriever_grader, str)
    assert isinstance(prompts_library.system_hallucination_grader, str)
    assert isinstance(prompts_library.system_answer_grader, str)
    assert isinstance(prompts_library.system_question_rewriter, str)
    assert isinstance(prompts_library.qa_prompt_template, str)
