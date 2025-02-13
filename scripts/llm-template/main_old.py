# main.py
import os
import wandb
from config import GOOGLE_API_KEY, WANDB_API_KEY, LANGSMITH_API_KEY, LANGCHAIN_PROJECT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import LangChainTracer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# Initialize Weights & Biases
wandb.login(key=WANDB_API_KEY)
run = wandb.init(project=LANGCHAIN_PROJECT, entity="aimingmed")

# Initialize Gemini API
tracer = LangChainTracer()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, callbacks=[tracer])

# Example usage of Gemini API
prompt_template = PromptTemplate(template="Write a short poem about the sun.", input_variables=[])
chain = LLMChain(llm=llm, prompt=prompt_template)
response = chain.run({})

# print(response)

import time
from langsmith import Client, Run
from langsmith.evaluation import EvaluationResult

# Initialize LangSmith client
client = Client(api_key=LANGSMITH_API_KEY)  # Replace with your API key
project_name = "my-gemini-evaluation"  # Your LangSmith project name

def evaluate_gemini_response(prompt, expected_response, gemini_response):
    """Evaluates Gemini's response against an expected response and logs to LangSmith."""

    # 1. Create a Run object (This is the correct way for manual logging)
    run = Run(
        client=client,
        project_name=project_name,
        inputs={"prompt": prompt, "expected_response": expected_response},  # Log inputs here
    )

    try:  # Use a try-except block for proper error handling
        start_time = time.time()
        
        # ... (Your Gemini API call goes here) ...
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", google_api_key=GOOGLE_API_KEY, callbacks=[tracer])
        # Example usage of Gemini API
        prompt_template = PromptTemplate(template="Start from here: ", input_variables=[])
        chain = LLMChain(llm=llm, prompt=prompt_template)
        gemini_response = chain.run({})  # Replace with your actual Gemini call

        end_time = time.time()
        latency = end_time - start_time

        run.outputs = {"gemini_response": gemini_response}  # Log outputs
        run.latency = latency # Log latency

        # 2. End the Run (Important!)
        run.end() # Mark the run as complete.

        # 3. Evaluate and log the result
        evaluation_result = evaluate_response(expected_response, gemini_response)
        run.create_evaluation(evaluation_result) # Log the evaluation

        return evaluation_result

    except Exception as e:  # Handle exceptions
        run.end(error=str(e))  # Log the error in LangSmith
        print(f"Error during Gemini call or evaluation: {e}")
        return None  # Or handle the error as needed

def evaluate_response(expected_response, gemini_response):
    """Performs the actual evaluation logic.  Customize this!"""

    # Example 1: Exact match (simple, but often not realistic)
    if expected_response.strip().lower() == gemini_response.strip().lower():
        score = 1.0  # Perfect match
        feedback = "Exact match!"
    # Example 2: Keyword overlap (more flexible)
    elif any(keyword in gemini_response.lower() for keyword in expected_response.lower().split()):
        score = 0.7  # Partial match (adjust score as needed)
        feedback = "Keyword overlap."
    # Example 3: Semantic similarity (requires external library/API) -  Advanced!
    # ... (Use a library like SentenceTransformers or an API for semantic similarity) ...
    else:
        score = 0.0
        feedback = "No match."

    # Create a LangSmith EvaluationResult object
    evaluation_result = EvaluationResult(
        score=score,
        value=gemini_response, # The actual response being evaluated
        comment=feedback,
        # Other metadata you might want to add, like prompt or expected response
        metadata = {"expected_response": expected_response} 
    )

    return evaluation_result



# Example usage:
prompt = "Translate 'Hello, world!' to French."
expected_response = "Bonjour le monde !"
gemini_response = "Bonjour monde !" # Or get the actual response from Gemini

evaluation = evaluate_gemini_response(prompt, expected_response, gemini_response)
print(f"Evaluation Score: {evaluation.score}, Feedback: {evaluation.comment}")

# Another example
prompt = "What is the capital of France?"
expected_response = "Paris"
gemini_response = "The capital of France is Paris."

evaluation = evaluate_gemini_response(prompt, expected_response, gemini_response)
print(f"Evaluation Score: {evaluation.score}, Feedback: {evaluation.comment}")

wandb.finish()