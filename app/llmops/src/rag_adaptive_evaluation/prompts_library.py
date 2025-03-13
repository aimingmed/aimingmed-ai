system_router = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to any cancer/tumor disease. The question may be
asked in a variety of languages, and may be phrased in a variety of ways.
Use the vectorstore for questions on these topics. Otherwise, use web-search. 
"""

system_retriever_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

system_hallucination_grader = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

system_answer_grader = """You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

system_question_rewriter = """You a question re-writer that converts an input question to a better version that is optimized \n 
    for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""


# Evaluation
CORRECTNESS_PROMPT = """Evaluate Student Answer against Ground Truth for conceptual similarity and correctness.

You are an impartial judge. Evaluate Student Answer against Ground Truth for conceptual similarity and correctness. 
You may also be given additional information that was used by the model to generate the output.

Your task is to determine a numerical score called faithfulness based on the input and output.
A definition of correctness and a grading rubric are provided below.
You must use the grading rubric to determine your score.

Metric definition:
Correctness assesses the degree to which a provided output aligns with factual accuracy, completeness, logical 
consistency, and precise terminology. It evaluates the intrinsic validity of the output, independent of any 
external context. A higher score indicates a higher adherence to factual accuracy, completeness, logical consistency, 
and precise terminology.

Grading rubric:
Correctness: Below are the details for different scores: 
 - 1: Major factual errors, highly incomplete, illogical, and uses incorrect terminology.
 - 2: Significant factual errors, incomplete, noticeable logical flaws, and frequent terminology errors.
 - 3: Minor factual errors, somewhat incomplete, minor logical inconsistencies, and occasional terminology errors.
 - 4: Few to no factual errors, mostly complete, strong logical consistency, and accurate terminology.
 - 5: Accurate, complete, logically consistent, and uses precise terminology.
 
 Reminder:
  - Carefully read the input and output
  - Check for factual accuracy and completeness
  - Focus on correctness of information rather than style or verbosity
  - The goal is to evaluate factual correctness and completeness of the response.
  - Please provide your answer score only with the numerical number between 1 and 5. No score: or other text is allowed.

"""

