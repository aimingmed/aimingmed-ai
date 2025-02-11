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

print(response)

wandb.finish()